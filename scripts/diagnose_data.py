import numpy as np
from pathlib import Path
import json
from collections import Counter

# Check processed data
data_dir = Path("data/processed/spectrograms")
metadata_files = list(data_dir.glob("*.json"))

print(f"Total processed files: {len(metadata_files)}")

# Check label distribution
labels = []
shapes = []

for metadata_path in metadata_files[:1000]:
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    labels.append(metadata["label"])
    
    processed_path = Path(metadata.get("processed_path"))
    if processed_path.exists():
        spec = np.load(processed_path)
        shapes.append(spec.shape)

print("\nLabel distribution:")
label_counts = Counter(labels)
for label, count in label_counts.most_common():
    print(f"  {label}: {count}")

print(f"\nSpectrogram shapes (first 10): {shapes[:10]}")

# Check for data quality
print("\nChecking first 5 spectrograms:")
for i, metadata_path in enumerate(metadata_files[:5]):
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    processed_path = Path(metadata.get("processed_path"))
    spec = np.load(processed_path)
    
    print(f"\n{metadata['label']}:")
    print(f"  Shape: {spec.shape}")
    print(f"  Min: {spec.min():.4f}, Max: {spec.max():.4f}, Mean: {spec.mean():.4f}")
    print(f"  Has NaN: {np.isnan(spec).any()}")
    print(f"  Has Inf: {np.isinf(spec).any()}")
    print(f"  Unique values: {len(np.unique(spec))}")
