import tensorflow as tf
import os

# --- NODE C: MODEL QUANTIZATION & OPTIMIZATION ---

# 1. Load the trained H5 model from Node B
model_path = 'har_model.h5'

if not os.path.exists(model_path):
    print(f"ERROR: {model_path} not found. Did you run 2_train_model.py successfully?")
else:
    print(f"Loading model for quantization: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # 2. Initialize the TFLite Converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # 3. Apply the "LSTM Fix" Flags
    # This allows TFLite to handle the complex 'TensorList' math used by LSTMs
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # Standard optimized ops
        tf.lite.OpsSet.SELECT_TF_OPS   # Support for complex LSTM kernels
    ]

    # Disable the experimental lowering that causes the "monotonically increasing" style error in TFLite
    converter._experimental_lower_tensor_list_ops = False
    
    # Enable basic optimization for file size
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    print("Converting model to TFLite format... (this may take a moment)")

    try:
        # 4. Perform the conversion
        tflite_model = converter.convert()

        # 5. Save the final optimized brain
        tflite_filename = 'fall_detection_model.tflite'
        with open(tflite_filename, 'wb') as f:
            f.write(tflite_model)

        print("-" * 30)
        print(f"SUCCESS: Node C Complete!")
        print(f"Optimized model saved as: {tflite_filename}")
        print("-" * 30)
        
    except Exception as e:
        print(f"\nQuantization Failed: {e}")