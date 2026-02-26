import streamlit as st
import pickle
import cv2
import numpy as np
from PIL import Image
import tempfile

# Load model dictionary
MODEL_PATH = "model/brain_tumor_model.pkl"
model_data = pickle.load(open(MODEL_PATH, "rb"))
model = model_data["model"]
model_accuracy = model_data["accuracy"]


def predict_tumor(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Invalid image file.")

    img = cv2.resize(img, (64, 64))
    img = img.flatten() / 255.0   # must match training
    img = img.reshape(1, -1)

    prediction = model.predict(img)[0]
    confidence = model.predict_proba(img)[0][prediction]

    return prediction, round(confidence * 100, 2)


# ---------------- STREAMLIT UI ---------------- #

st.title("Brain Tumor Detection")

st.write(f"**Model Accuracy:** {round(model_accuracy * 100, 2)}%")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:

    st.image(uploaded_file, caption="Uploaded MRI", use_container_width=True)

    # Save image temporarily (same as Flask saving)
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    prediction, confidence = predict_tumor(temp_path)

    if prediction == 1:
        result = "🧠 Brain Tumor Detected"
        suggestion = "Consult a neurologist immediately."
    else:
        result = "✅ No Tumor Detected"
        suggestion = "Maintain regular health checkups."

    st.subheader(result)
    st.write(f"**Confidence:** {confidence}%")
    st.write(f"**Suggestion:** {suggestion}")
