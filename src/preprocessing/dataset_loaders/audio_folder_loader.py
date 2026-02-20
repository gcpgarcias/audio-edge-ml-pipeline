"""
Generic audio folder dataset loader.

Handles the standard ``class-per-subfolder`` layout used by most audio
classification benchmarks (e.g. ESC-50, UrbanSound8k, Google Speech Commands):

    <dataset_root>/
        class_a/
            clip_001.wav
            clip_002.flac
        class_b/
            clip_001.mp3
        …

Optionally a split subdirectory (``train``, ``test``, ``val``) can sit
between the root and the class directories:

    <dataset_root>/
        train/
            class_a/ …
            class_b/ …
        test/
            class_a/ …

Metadata
--------
Each sample yields::

    (audio_path, class_label, {
        "filename":    str,
        "class_dir":   str,
        "duration":    float,   # seconds
        "sample_rate": int,     # Hz
        "n_channels":  int,
    })

Metadata is read at initialisation using ``soundfile.info()`` — a header-only
probe that does not decode the audio.  For formats unsupported by soundfile
(e.g. MP3, AAC) the numeric fields default to ``0`` / ``0.0``.

Usage
-----
::

    from src.preprocessing.dataset_loaders.audio_folder_loader import AudioFolderLoader
    from src.preprocessing.feature_extraction import get

    loader    = AudioFolderLoader("path/to/dataset", split="train")
    extractor = get("audio_classical")()
    fs        = extractor.extract_dataset(loader)

Compared with :class:`BIRDeepLoader`
-------------------------------------
:class:`BIRDeepLoader` is specific to the BIRDeep_AudioAnnotations CSV schema
(annotated segments, species labels, bbox metadata).  :class:`AudioFolderLoader`
is the generic counterpart for any unlabelled or class-labelled audio collection
stored in the class-per-folder convention.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

from ..feature_extraction.base import BaseDatasetLoader

logger = logging.getLogger(__name__)

# Audio file extensions recognised by the loader (case-insensitive).
_AUDIO_SUFFIXES: frozenset[str] = frozenset(
    {".wav", ".flac", ".ogg", ".mp3", ".aac", ".m4a", ".opus", ".aiff", ".aif"}
)


def _audio_meta(path: Path) -> dict:
    """Probe audio file metadata without decoding samples.

    Uses ``soundfile.info()`` for formats it natively supports (WAV, FLAC,
    OGG, CAF, AIFF, …).  Falls back to zero-valued defaults for unsupported
    formats (MP3, AAC) rather than raising.

    Parameters
    ----------
    path:
        Path to an audio file.

    Returns
    -------
    dict
        Keys: ``duration`` (float, seconds), ``sample_rate`` (int, Hz),
        ``n_channels`` (int).  Values are 0 / 0.0 on failure.
    """
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return {
            "duration":    info.duration,
            "sample_rate": info.samplerate,
            "n_channels":  info.channels,
        }
    except Exception:
        return {"duration": 0.0, "sample_rate": 0, "n_channels": 0}


class AudioFolderLoader(BaseDatasetLoader):
    """Load audio files from a class-per-subfolder directory tree.

    Parameters
    ----------
    root:
        Dataset root directory.  Class labels are taken from the names of
        its immediate subdirectories (or from ``<root>/<split>/`` when
        *split* is given).
    split:
        Optional split subdirectory name (``"train"``, ``"test"``,
        ``"val"``, etc.).  When provided, ``<root>/<split>/`` is used as
        the effective root.
    extensions:
        Set of file extensions to include (case-insensitive, with leading
        dot).  Defaults to common audio formats.
    class_names:
        If given, only files belonging to these class folders are loaded,
        in this order.  Unknown classes are silently skipped.  When *None*,
        all subdirectories are treated as class folders (sorted
        alphabetically for reproducibility).
    """

    def __init__(
        self,
        root:        Path | str,
        split:       Optional[str]       = None,
        extensions:  Optional[set[str]]  = None,
        class_names: Optional[list[str]] = None,
    ) -> None:
        effective_root = Path(root) / split if split else Path(root)
        if not effective_root.is_dir():
            raise NotADirectoryError(
                f"Dataset root not found: {effective_root}"
            )

        self._extensions = (
            frozenset(e.lower() for e in extensions)
            if extensions is not None
            else _AUDIO_SUFFIXES
        )

        # Discover class folders
        if class_names is not None:
            self._class_names = list(class_names)
            class_dirs = [effective_root / c for c in class_names]
        else:
            class_dirs = sorted(
                p for p in effective_root.iterdir() if p.is_dir()
            )
            self._class_names = [d.name for d in class_dirs]

        # Build flat list of (audio_path, class_label, meta_dict)
        self._samples: list[tuple[Path, str, dict]] = []
        for class_dir, label in zip(class_dirs, self._class_names):
            if not class_dir.is_dir():
                logger.warning("Class directory not found: %s (skipping)", class_dir)
                continue
            clips = sorted(
                p for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in self._extensions
            )
            if not clips:
                logger.warning("No audio files found in: %s", class_dir)
            for clip_path in clips:
                meta = {
                    "filename":  clip_path.name,
                    "class_dir": class_dir.name,
                    **_audio_meta(clip_path),
                }
                self._samples.append((clip_path, label, meta))

        logger.info(
            "AudioFolderLoader: %d clips across %d classes.",
            len(self._samples),
            len(self._class_names),
        )

    # ------------------------------------------------------------------
    # BaseDatasetLoader interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[tuple[Path, Optional[str], dict]]:
        for audio_path, label, meta in self._samples:
            yield audio_path, label, meta

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def class_names(self) -> list[str]:
        """Alphabetically sorted class labels discovered in the root."""
        return list(self._class_names)

    @property
    def n_classes(self) -> int:
        return len(self._class_names)