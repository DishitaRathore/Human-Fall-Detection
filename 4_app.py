import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- PAGE CONFIG ---
st.set_page_config(page_title="Real-Time Fall Detection", layout="wide")
st.title("🤖 Live Robotics Perception Hub")

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO
    )
    detector = vision.PoseLandmarker.create_from_options(options)
    model = tf.keras.models.load_model('har_model.h5')
    return detector, model

detector, model = load_models()

# --- INITIALIZE VARIABLES ---
run = st.checkbox('Start Live Monitoring', value=True)
frame_window = st.image([]) 
status_msg = st.empty()

# --- THE FIX: Initialize these lists so they exist when the loop starts ---
if 'buffer' not in st.session_state:
    st.session_state.buffer = []
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# --- THE LIVE LOOP ---
cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret: break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
    result = detector.detect_for_video(mp_image, timestamp)
    
    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]
        frame_features = []
        
        # Node A: Feature Extraction
        h, w, _ = frame.shape
        for lm in landmarks:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, (0, 255, 0), -1)
            frame_features.extend([lm.x, lm.y, lm.z, lm.presence])
        
        # Update Buffer
        st.session_state.buffer.append(frame_features)
        if len(st.session_state.buffer) > 30:
            st.session_state.buffer.pop(0)
            
        # Node B: Intelligence Prediction
        if len(st.session_state.buffer) == 30:
            input_data = np.expand_dims(st.session_state.buffer, axis=0)
            prediction = model.predict(input_data, verbose=0)[0][0]
            
            # Store prediction in history
            st.session_state.prediction_history.append(prediction)
            if len(st.session_state.prediction_history) > 10:
                st.session_state.prediction_history.pop(0)
            
            # --- MOVING AVERAGE LOGIC ---
            # We average the last 5 predictions to filter out "noise"
            avg_pred = np.mean(st.session_state.prediction_history[-5:])
            
            if avg_pred > 0.8:
                cv2.putText(frame, "!!! FALL DETECTED !!!", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                status_msg.error(f"🚨 CONFIRMED FALL (Confidence: {avg_pred:.2f})")
            else:
                status_msg.success("✅ Monitoring... Normal Activity")
    else:
        status_msg.warning("⚠️ Adjusting: Searching for Person...")

    frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()