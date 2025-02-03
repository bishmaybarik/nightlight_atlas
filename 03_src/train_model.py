import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import os

# Load dataset
data_path = "/Users/bishmaybarik/Library/CloudStorage/OneDrive-ShivNadarInstitutionofEminence/nightlight_atlas/01_data/02_processed/secc_combined_updated.csv"
df = pd.read_csv(data_path)

# Define input features
X = df[["dmsp_scaled", "urban_dummy", "nightlight_area_interaction"]].values
y = np.log1p(df["monthly_per_capita_consumption"].values)  # Log transformation

# Split data into training and testing sets (80% train, 20% test)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a simple feedforward neural network
model = keras.Sequential([
    layers.Dense(16, activation="relu", input_shape=(X.shape[1],)),  # First hidden layer
    layers.Dense(8, activation="relu"),  # Second hidden layer
    layers.Dense(1, activation="linear")  # Output layer (continuous value)
])

# Compile the model
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Save the model
model_save_path = "/Users/bishmaybarik/Library/CloudStorage/OneDrive-ShivNadarInstitutionofEminence/nightlight_atlas/03_src/model.h5"
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
model.save(model_save_path)

print(f"Model saved to {model_save_path}")
