import sys
sys.path.append('.')
import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
from src.training.dataset import SpectrogramDataset

# Load data
print("Loading dataset...")
dataset_loader = SpectrogramDataset(Path("data/processed/spectrograms"))
train_ds, val_ds, class_names = dataset_loader.load_dataset()

train_ds = dataset_loader.prepare_for_training(train_ds, batch_size=32, shuffle=True)
val_ds = dataset_loader.prepare_for_training(val_ds, batch_size=32, shuffle=False)

# Get a batch to check input shape
for X, y in train_ds.take(1):
    input_shape = X.shape[1:]
    print(f"Input shape: {input_shape}")

# Create extremely simple model
print("\nCreating ultra-simple model...")
model = keras.Sequential([
    keras.layers.Flatten(input_shape=input_shape),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(5, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train for just a few epochs
print("\nTraining ultra-simple model...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    verbose=1
)

print(f"\nFinal training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

# Test predictions
print("\nChecking predictions on validation set...")
for X_batch, y_batch in val_ds.take(1):
    predictions = model.predict(X_batch, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    
    print(f"True labels: {y_batch.numpy()[:10]}")
    print(f"Predictions: {predicted_classes[:10]}")
    print(f"Predicted probs (first sample): {predictions[0]}")
    
    correct = np.sum(predicted_classes == y_batch.numpy())
    print(f"Correct in batch: {correct}/{len(y_batch)} ({100*correct/len(y_batch):.1f}%)")
