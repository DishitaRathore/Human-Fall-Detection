import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = 'pose_landmarker_heavy.task' 

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.PoseLandmarker.create_from_options(options)

DATA_DIR = "./Dataset" 
CATEGORIES = ["Daily", "Fall"]
dataset = []

global_timestamp_ms = 0

print("--- Node A: Modern Perception Hub Started ---")

for category in CATEGORIES:
    class_num = CATEGORIES.index(category)
    path = os.path.join(DATA_DIR, category)
    
    if not os.path.exists(path):
        print(f"Skipping {category}: Folder not found.")
        continue

    for sequence_folder in os.listdir(path):
        seq_path = os.path.join(path, sequence_folder)
        if not os.path.isdir(seq_path): continue
            
        print(f"Processing sequence: {sequence_folder}")
        sequence_landmarks = []
        
        # Sort images to ensure temporal order
        images = sorted([i for i in os.listdir(seq_path) if i.endswith(".png")])
        
        for img_name in images:
            img_path = os.path.join(seq_path, img_name)
            image = cv2.imread(img_path)
            if image is None: continue
           
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            
      
            detection_result = detector.detect_for_video(mp_image, global_timestamp_ms)
            
            if detection_result.pose_landmarks:
               
                landmarks = detection_result.pose_landmarks[0]
                frame_features = []
                for lm in landmarks:
                    frame_features.extend([lm.x, lm.y, lm.z, lm.presence])
                sequence_landmarks.append(frame_features)
            
            global_timestamp_ms += 33 
        
        if len(sequence_landmarks) >= 30:
            dataset.append([sequence_landmarks[:30], class_num])
        else:
            print(f"Warning: {sequence_folder} had only {len(sequence_landmarks)} valid frames. Skipping.")

if dataset:
    X = np.array([item[0] for item in dataset])
    y = np.array([item[1] for item in dataset])
    np.save("X_data.npy", X)
    np.save("y_data.npy", y)
    print(f"\nSUCCESS! Processed {len(dataset)} total sequences.")
    print("Files saved: X_data.npy (Features) and y_data.npy (Labels)")
else:
    print("\nERROR: No sequences were processed. Check your dataset folder structure.")