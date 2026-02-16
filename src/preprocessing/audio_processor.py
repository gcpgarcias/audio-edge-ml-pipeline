"""
Ingestion-layer audio preprocessor.

Converts raw audio files (uploaded via the ingestion API) to log-mel
spectrograms and persists them alongside their JSON metadata.

This module is kept intentionally thin: the actual feature computation is
delegated to :class:`~src.preprocessing.feature_extraction.audio.deep.AudioMelSpectrogram`
so that parameters stay consistent across the ingestion path and the
training pipeline.
"""

import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np

from src.preprocessing.feature_extraction.audio.deep import AudioMelSpectrogram

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Batch-process ingested audio files into stored mel-spectrograms.

    Parameters mirror those of :class:`AudioMelSpectrogram` so that the
    representation written to disk is identical to what the training
    pipeline reads back via the feature extraction layer.
    """

    def __init__(
        self,
        sample_rate: int   = 16000,
        n_mels:      int   = 40,
        n_fft:       int   = 512,
        hop_length:  int   = 160,
        duration:    float = 1.0,
    ) -> None:
        self._extractor = AudioMelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            duration=duration,
        )

    # Preserve the original public method name used by the ingestion API.
    def load_and_preprocess(self, audio_path: Path) -> np.ndarray:
        """Load *audio_path* and return a normalized log-mel spectrogram.

        Returns
        -------
        np.ndarray
            Shape ``(n_mels, time_frames)``, dtype float32, values in [0, 1].
        """
        return self._extractor.extract(audio_path)

    def process_dataset(self, input_dir: Path, output_dir: Path) -> Dict[str, int]:
        """Process all JSON-annotated audio files in *input_dir*.

        Reads each ``*.json`` metadata file, extracts features from the
        referenced audio, saves a ``.npy`` file, and updates the metadata.

        Returns
        -------
        dict
            ``{"processed": N, "failed": M}``
        """
        input_dir  = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stats = {"processed": 0, "failed": 0}

        for metadata_path in input_dir.glob("*.json"):
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)

                audio_path = Path(metadata["path"])
                if not audio_path.exists():
                    logger.warning("Audio file not found: %s", audio_path)
                    stats["failed"] += 1
                    continue

                spec = self.load_and_preprocess(audio_path)

                file_id     = metadata["file_id"]
                output_path = output_dir / f"{file_id}.npy"
                np.save(output_path, spec)

                metadata["processed_path"]   = str(output_path)
                metadata["spectrogram_shape"] = list(spec.shape)

                out_meta = output_dir / f"{file_id}.json"
                with open(out_meta, "w") as f:
                    json.dump(metadata, f, indent=2)

                stats["processed"] += 1
                if stats["processed"] % 50 == 0:
                    logger.info("Processed %d files…", stats["processed"])

            except Exception as exc:
                logger.error("Error processing %s: %s", metadata_path, exc)
                stats["failed"] += 1

        logger.info("Processing complete: %s", stats)
        return stats


def main() -> None:
    preprocessor = AudioPreprocessor()
    stats = preprocessor.process_dataset(
        input_dir=Path("data/raw/uploads"),
        output_dir=Path("data/processed/spectrograms"),
    )
    print(f"Preprocessing complete: {stats}")


if __name__ == "__main__":
    main()