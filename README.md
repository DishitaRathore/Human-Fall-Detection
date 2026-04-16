# FallGuard AI: Real-Time Human Fall Detection System

FallGuard AI is a comprehensive robotics perception project designed to detect human falls in real-time. By utilizing **MediaPipe Pose Landmarking** and **LSTM (Long Short-Term Memory)** neural networks, the system analyzes human motion sequences to distinguish between daily activities and emergency fall events.

## 🚀 Key Features
- **Privacy-First Tracking**: Uses 33 body landmarks (skeletons) instead of raw video pixels to ensure user privacy.
- **Temporal Intelligence**: Employs an LSTM model to understand motion over time (30-frame windows).
- **Edge Optimized**: Includes a quantization pipeline to convert models to TFLite for deployment on low-power devices.
- **Instant Notifications**: Integrated with Telegram Bot API for real-time mobile alerts.
- **Live Monitoring**: Streamlit-based dashboard for real-time visualization and status reporting.

## 🛠️ System Architecture (The 4 Nodes)

### Node A: Data Extraction (`1_extract_data.py`)
Processes raw image sequences into structured mathematical data.
- **Tools**: MediaPipe, OpenCV.
- **Output**: `X_data.npy` (Features) and `y_data.npy` (Labels).

### Node B: Intelligence Hub (`2_train_model.py`)
Trains the "brain" of the system using deep learning.
- **Model**: Sequential LSTM with Dropout layers to prevent overfitting.
- **Technique**: Data Augmentation (Horizontal Flipping) to double training samples.

### Node C: Optimization (`3_quantize_model.py`)
Optimizes the model for robotics and edge computing.
- **Conversion**: Converts `.h5` Keras models to `.tflite`.
- **Optimization**: Applies quantization to reduce file size and increase inference speed.

### Node D: Deployment Hub (`4_app.py`)
The live interface for monitoring and alerting.
- **Framework**: Streamlit.
- **Logic**: Moving average prediction for high-accuracy detection and Telegram alerts.

## 📂 Dataset
The project is built and tested using the **UR Fall Detection Dataset**. Due to file size constraints, the raw dataset is not included in this repository.

- **Official Source**: [UR Fall Detection Dataset](https://fenix.ur.edu.pl/mkepski/ds/uf.html)
- **Kaggle Mirror**: [UR Fall Detection on Kaggle](https://www.kaggle.com/datasets/shahliza27/ur-fall-detection-dataset)

**Note**: Ensure your dataset folder structure matches `./Dataset/Daily` and `./Dataset/Fall` before running Node A.

## ⚙️ Installation & Usage

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/yourusername/fallguard-ai.git](https://github.com/yourusername/fallguard-ai.git)
   cd fallguard-ai
