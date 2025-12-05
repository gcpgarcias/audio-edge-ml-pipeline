import tensorflow as tf
import numpy as np
from pathlib import Path
import json
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)

class SpectrogramDataset:
    def __init__(self, data_dir: Path, validation_split: float = 0.2):
        self.data_dir = Path(data_dir)
        self.validation_split = validation_split
        self.class_names = []
        self.num_classes = 0
    
    def load_dataset(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str]]:
        """Load processed spectrograms and create train/val datasets."""
        # Collect all processed files
        metadata_files = list(self.data_dir.glob("*.json"))
        
        spectrograms = []
        labels = []
        
        for metadata_path in metadata_files:
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            processed_path = Path(metadata.get("processed_path"))
            if not processed_path.exists():
                continue
            
            label = metadata["label"]
            
            # Load spectrogram
            spectrogram = np.load(processed_path)
            spectrograms.append(spectrogram)
            labels.append(label)
        
        # Create class mapping
        unique_labels = sorted(set(labels))
        self.class_names = unique_labels
        self.num_classes = len(unique_labels)
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        
        # Convert to arrays
        X = np.array(spectrograms)
        y = np.array([label_to_idx[label] for label in labels])
        
        # Add channel dimension for CNN
        X = X[..., np.newaxis]
        
        # Shuffle and split
        indices = np.random.permutation(len(X))
        split_idx = int(len(X) * (1 - self.validation_split))
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        logger.info(f"Dataset loaded: {len(X_train)} train, {len(X_val)} val samples")
        logger.info(f"Classes: {self.class_names}")
        logger.info(f"Spectrogram shape: {X_train[0].shape}")
        
        # Create TensorFlow datasets
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        
        return train_ds, val_ds, self.class_names
    
    def prepare_for_training(
        self,
        dataset: tf.data.Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        augment: bool = False
    ) -> tf.data.Dataset:
        """Prepare dataset for training with batching and prefetching."""
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
