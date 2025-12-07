from pathlib import Path
import json

# Check raw uploads
uploads_dir = Path("data/raw/uploads")
wav_files = list(uploads_dir.glob("*.wav"))
json_files = list(uploads_dir.glob("*.json"))

print(f"Raw uploads:")
print(f"  WAV files: {len(wav_files)}")
print(f"  JSON metadata: {len(json_files)}")

# Check processed
processed_dir = Path("data/processed/spectrograms")
processed_json = list(processed_dir.glob("*.json"))
processed_npy = list(processed_dir.glob("*.npy"))

print(f"\nProcessed spectrograms:")
print(f"  NPY files: {len(processed_npy)}")
print(f"  JSON metadata: {len(processed_json)}")

# Check first few uploads metadata
print(f"\nFirst 5 upload metadata files:")
for json_file in json_files[:5]:
    with open(json_file) as f:
        metadata = json.load(f)
    print(f"  {json_file.name}: label='{metadata.get('label', 'MISSING')}'")

# Check first few processed metadata
print(f"\nFirst 5 processed metadata files:")
for json_file in processed_json[:5]:
    with open(json_file) as f:
        metadata = json.load(f)
    label = metadata.get('label', 'MISSING')
    processed_path = metadata.get('processed_path', 'MISSING')
    print(f"  {json_file.name}: label='{label}', has_processed={Path(processed_path).exists() if processed_path != 'MISSING' else False}")

# Check dataset loading
print(f"\nChecking dataset.py loading logic...")
from src.training.dataset import SpectrogramDataset

dataset_loader = SpectrogramDataset(processed_dir)
train_ds, val_ds, class_names = dataset_loader.load_dataset()

print(f"Dataset loader found:")
print(f"  Train samples: {len(list(train_ds))}")
print(f"  Val samples: {len(list(val_ds))}")
print(f"  Classes: {class_names}")
