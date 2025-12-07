import sys
sys.path.append('.')
import numpy as np
from pathlib import Path
from src.training.dataset import SpectrogramDataset
import tensorflow as tf

# Load dataset exactly as training does
dataset_loader = SpectrogramDataset(Path("data/processed/spectrograms"))
train_ds, val_ds, class_names = dataset_loader.load_dataset()

print(f"Classes: {class_names}")
print(f"Num classes: {len(class_names)}")

# Check first training batch
train_ds_batched = dataset_loader.prepare_for_training(train_ds, batch_size=32, shuffle=False)

for X_batch, y_batch in train_ds_batched.take(1):
    print(f"\nFirst training batch:")
    print(f"  X shape: {X_batch.shape}")
    print(f"  X dtype: {X_batch.dtype}")
    print(f"  X min: {X_batch.numpy().min():.4f}, max: {X_batch.numpy().max():.4f}, mean: {X_batch.numpy().mean():.4f}")
    print(f"  X has NaN: {np.isnan(X_batch.numpy()).any()}")
    print(f"  X has Inf: {np.isinf(X_batch.numpy()).any()}")
    
    print(f"\n  y shape: {y_batch.shape}")
    print(f"  y dtype: {y_batch.dtype}")
    print(f"  y unique values: {np.unique(y_batch.numpy())}")
    print(f"  y distribution in batch: {np.bincount(y_batch.numpy())}")
    
    # Check a single spectrogram
    sample = X_batch[0].numpy()
    print(f"\n  Single spectrogram stats:")
    print(f"    Shape: {sample.shape}")
    print(f"    Unique values: {len(np.unique(sample))}")
    print(f"    Value range: [{sample.min():.4f}, {sample.max():.4f}]")
    print(f"    Std dev: {sample.std():.4f}")

# Check validation data
val_ds_batched = dataset_loader.prepare_for_training(val_ds, batch_size=32, shuffle=False)

for X_batch, y_batch in val_ds_batched.take(1):
    print(f"\nFirst validation batch:")
    print(f"  X shape: {X_batch.shape}")
    print(f"  y shape: {y_batch.shape}")
    print(f"  y unique values: {np.unique(y_batch.numpy())}")
    print(f"  y distribution: {np.bincount(y_batch.numpy())}")

# Check if labels are consistent
print(f"\nLabel to index mapping:")
for i, class_name in enumerate(class_names):
    print(f"  {i}: {class_name}")
