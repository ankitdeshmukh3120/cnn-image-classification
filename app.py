import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image

# Page setup
st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐾", layout="centered")

st.title("🐾 Cat vs Dog Image Classifier")
st.write(
    "Upload an image of a cat or a dog, and this CNN model (trained from scratch with "
    "TensorFlow/Keras, 90.9% test accuracy) will predict which one it is."
)

# Load model (cached so it only loads once per session)
@st.cache_resource
def load_model():
    return keras.models.load_model("cats_vs_dogs_cnn.keras")

model = load_model()

# Matches the input size used during training
IMG_SIZE = (128, 128)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_resized = image.resize(IMG_SIZE)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner("Classifying..."):
        prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        label = "Dog"
        confidence = prediction * 100
    else:
        label = "Cat"
        confidence = (1 - prediction) * 100

    st.subheader(f"Prediction: **{label}** 🎯")
    st.write(f"Confidence: {confidence:.2f}%")
    st.progress(int(confidence))

st.markdown("---")
st.caption("Built by Ankit Deshmukh | [GitHub](https://github.com/ankitdeshmukh3120)")
