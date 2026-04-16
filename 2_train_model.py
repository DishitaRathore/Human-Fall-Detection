import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# 1. Load the data you extracted in Node A
X = np.load("X_data.npy")
y = np.load("y_data.npy")

print(f"Original shape: {X.shape}")

# --- DATA AUGMENTATION: HORIZONTAL FLIP ---
def flip_data(X_data):
    X_flipped = X_data.copy()
    # Every 4th value starting from 0 is an X-coordinate (x, y, z, presence)
    for s in range(X_flipped.shape[0]): # For each sequence
        for f in range(X_flipped.shape[1]): # For each frame
            for i in range(0, 132, 4): # For each joint's X-coord
                X_flipped[s, f, i] = 1.0 - X_flipped[s, f, i]
    return X_flipped

X_aug = flip_data(X)

# Combine original and flipped data (Doubles the dataset)
X_final = np.concatenate((X, X_aug), axis=0)
y_final = np.concatenate((y, y), axis=0)

print(f"Augmented shape: {X_final.shape} (Dataset Doubled!)")

# 2. Split into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2)

# 3. Build Node B: The Intelligence Hub (LSTM)
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(30, 132)),
    Dropout(0.2),
    LSTM(128, return_sequences=False),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid') # Binary Output: 0 (Daily) or 1 (Fall)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Train
print("Starting Training...")
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# 5. Save the upgraded brain
model.save("har_model.h5")
print("SUCCESS: Upgraded model saved as har_model.h5")