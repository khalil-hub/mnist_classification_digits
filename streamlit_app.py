import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from src.data_loader import pre_process_real_image
import cv2

# Load model
model = load_model("models/non_augmented/mnist_cnn_adam_10.h5")

st.title(" MNIST Digit Classifier")
st.write("Upload a handwritten digit image (white background preferred)")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    st.image(image, caption="Uploaded Image", width=150)

    # Save temporarily and preprocess
    cv2.imwrite("temp_input.jpg", image)
    x_input = pre_process_real_image("temp_input.jpg")

    # Predict
    pred = model.predict(x_input)
    predicted_class = np.argmax(pred)
    confidence = np.max(pred)

    st.markdown(f"### Predicted Digit: **{predicted_class}**")
    st.markdown(f"Confidence: `{confidence:.2f}`")
