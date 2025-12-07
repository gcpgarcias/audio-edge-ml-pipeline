import tensorflow as tf
import numpy as np
from pathlib import Path
import json

# Load model
model = tf.keras.models.load_model("data/models/audio_classifier.h5")

# Load some test data
data_dir = Path("data/processed/spectrograms")
metadata_files = list(data_dir.glob("*.json"))[:10]

print("Testing original Keras model:")
print("="*60)

for metadata_path in metadata_files:
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    processed_path = Path(metadata.get("processed_path"))
    if not processed_path.exists():
        continue
    
    spectrogram = np.load(processed_path)
    label = metadata["label"]
    
    # Add batch and channel dimensions
    input_data = np.expand_dims(spectrogram, axis=0)
    input_data = np.expand_dims(input_data, axis=-1)
    
    # Predict
    predictions = model.predict(input_data, verbose=0)
    probabilities = tf.nn.softmax(predictions[0]).numpy()
    
    predicted_idx = np.argmax(probabilities)
    confidence = probabilities[predicted_idx]
    
    # Load class names
    with open("data/models/class_names.json") as f:
        class_names = json.load(f)
    
    predicted_class = class_names[predicted_idx]
    
    print(f"True: {label:6s} | Predicted: {predicted_class:6s} | Confidence: {confidence:.3f}")
    print(f"  All probs: {probabilities}")
