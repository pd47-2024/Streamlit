import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("angle_classification_model.h5")

# Define your class labels (assuming you have these defined somewhere in the script)
class_labels = ["15-18deg", "27-30deg"]

# Prediction function
def predict_image(image):
    img = cv2.resize(image, (64, 64)) / 255.0  # Resize and normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    predictions = model.predict(img)
    class_idx = np.argmax(predictions)
    return class_labels[class_idx]

# Streamlit file uploader
uploaded_file = st.file_uploader("Upload an image for classification", type=["jpg", "png"])
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, channels="RGB")
    
    # Predict and display result
    predicted_class = predict_image(img)
    st.write("Predicted Class:", predicted_class)
