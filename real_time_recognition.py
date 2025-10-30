import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from tensorflow.keras.models import load_model
import pickle

# Load Models
try:
    cnn_model = load_model("gesture_cnn_model.h5")
    with open("gesture_rf_model.pkl", "rb") as f:
        rf_model = pickle.load(f)
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# Label mapping A–Z
labels = [chr(i) for i in range(65, 91)]

# Streamlit UI
st.title("🖐️ Real-Time ASL Gesture Recognition (CNN + Random Forest)")

# Reference images 
st.image(["american_sign_language.PNG", "amer_sign2.png", "amer_sign3.png"],
         caption=["ASL Reference 1", "ASL Reference 2", "ASL Reference 3"], width=500)

# Helper function for preprocessing
def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, 28, 28, 1)
    return reshaped

# Video Processing Class
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        processed = preprocess(img)

        # CNN feature extraction
        cnn_features = cnn_model.predict(processed)
        
        # RF classification
        rf_pred = rf_model.predict(cnn_features.reshape(1, -1))
        label = labels[int(rf_pred[0])] if int(rf_pred[0]) < len(labels) else "?"

        # Display prediction
        cv2.putText(img, f'Prediction: {label}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
        return img

# Start Webcam Stream
webrtc_streamer(key="gesture", video_processor_factory=VideoProcessor)

st.markdown("👆 Show your hand gestures (A–Z) to the webcam for real-time recognition! "
            "Ensure your hand is well-lit and visible.")
st.markdown("Developed using a CNN for feature extraction and a Random Forest classifier for gesture recognition.")