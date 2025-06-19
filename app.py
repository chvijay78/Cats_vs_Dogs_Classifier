import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Set page config
st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐾")

# Load model
model = tf.keras.models.load_model('cats_vs_dogs_cnn.h5')

# Preprocessing function
def preprocess_image(image):
    image = image.resize((160, 160))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Title and description
st.title("🐶 Cat vs 🐱 Dog Classifier")
st.markdown("Upload an image and the AI will predict if it's a cat or a dog.")

# Upload image
uploaded_file = st.file_uploader("Choose a file...")

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing the image..."):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)[0][0]
        confidence = round(prediction * 100, 2) if prediction > 0.5 else round((1 - prediction) * 100, 2)

        if prediction > 0.5:
            st.success(f"🐶 It's a **Dog** with {confidence}% confidence.")
        else:
            st.success(f"🐱 It's a **Cat** with {confidence}% confidence.")
