import librosa
import numpy as np
from pathlib import Path
import json
from typing import Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioPreprocessor:
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 40,
        n_fft: int = 512,
        hop_length: int = 160,
        duration: float = 1.0
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.duration = duration
        self.target_length = int(sample_rate * duration)
    
    def load_and_preprocess(self, audio_path: Path) -> np.ndarray:
        """Load audio file and convert to log mel spectrogram."""
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
        
        # Pad or trim to target length
        if len(audio) < self.target_length:
            audio = np.pad(audio, (0, self.target_length - len(audio)))
        else:
            audio = audio[:self.target_length]
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1]
        log_mel_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min() + 1e-8)
        
        return log_mel_spec
    
    def process_dataset(self, input_dir: Path, output_dir: Path) -> Dict[str, int]:
        """Process all audio files in directory structure."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stats = {"processed": 0, "failed": 0}
        
        # Find all audio files with metadata
        metadata_files = list(input_dir.glob("*.json"))
        
        for metadata_path in metadata_files:
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
                
                audio_path = Path(metadata["path"])
                if not audio_path.exists():
                    logger.warning(f"Audio file not found: {audio_path}")
                    stats["failed"] += 1
                    continue
                
                # Process audio
                log_mel_spec = self.load_and_preprocess(audio_path)
                
                # Save processed features
                file_id = metadata["file_id"]
                output_path = output_dir / f"{file_id}.npy"
                np.save(output_path, log_mel_spec)
                
                # Update metadata with processed path
                metadata["processed_path"] = str(output_path)
                metadata["spectrogram_shape"] = log_mel_spec.shape
                
                output_metadata_path = output_dir / f"{file_id}.json"
                with open(output_metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                
                stats["processed"] += 1
                
                if stats["processed"] % 50 == 0:
                    logger.info(f"Processed {stats['processed']} files...")
                
            except Exception as e:
                logger.error(f"Error processing {metadata_path}: {e}")
                stats["failed"] += 1
        
        logger.info(f"Processing complete: {stats}")
        return stats

def main():
    preprocessor = AudioPreprocessor()
    
    input_dir = Path("data/raw/uploads")
    output_dir = Path("data/processed/spectrograms")
    
    stats = preprocessor.process_dataset(input_dir, output_dir)
    print(f"Preprocessing complete: {stats}")

if __name__ == "__main__":
    main()
