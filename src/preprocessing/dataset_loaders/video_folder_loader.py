"""
Generic video folder dataset loader.

Handles the standard ``class-per-subfolder`` layout:

    <dataset_root>/
        class_a/
            clip_001.mp4
            clip_002.avi
        class_b/
            clip_001.mp4
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

    (video_path, class_label, {
        "filename": str,
        "class_dir": str,
        "n_frames": int,   # total frame count reported by the container
        "fps":      float, # frames-per-second
        "duration": float, # duration in seconds
    })

The ``n_frames``, ``fps``, and ``duration`` fields are read at initialisation
time using ``cv2.VideoCapture`` property queries — no actual frames are decoded
during loader construction.

Usage
-----
::

    from src.preprocessing.dataset_loaders.video_folder_loader import VideoFolderLoader
    from src.preprocessing.feature_extraction import get

    loader    = VideoFolderLoader("path/to/dataset", split="train")
    extractor = get("video_classical")()
    fs = extractor.extract_dataset(loader)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

from ..feature_extraction.base import BaseDatasetLoader

logger = logging.getLogger(__name__)

# Video file extensions recognised by the loader (case-insensitive).
_VIDEO_SUFFIXES: frozenset[str] = frozenset(
    {".mp4", ".avi", ".mov", ".mkv", ".webm"}
)


def _video_meta(path: Path) -> dict:
    """Query basic video metadata without decoding frames.

    Parameters
    ----------
    path:
        Path to a video file.

    Returns
    -------
    dict
        Keys: ``n_frames`` (int), ``fps`` (float), ``duration`` (float).
        Values are 0 / 0.0 if the container cannot report them.
    """
    import cv2  # lazy import — opencv-python must be installed

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        logger.warning("Cannot open video for metadata query: %s", path)
        cap.release()
        return {"n_frames": 0, "fps": 0.0, "duration": 0.0}

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps      = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    duration = (n_frames / fps) if fps > 0 else 0.0
    cap.release()

    return {"n_frames": n_frames, "fps": fps, "duration": duration}


class VideoFolderLoader(BaseDatasetLoader):
    """Load videos from a class-per-subfolder directory tree.

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
        dot).  Defaults to ``.mp4 .avi .mov .mkv .webm``.
    class_names:
        If given, only videos belonging to these class folders are loaded,
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
            else _VIDEO_SUFFIXES
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

        # Build flat list of (video_path, class_label, meta_dict)
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
                logger.warning("No video files found in: %s", class_dir)
            for clip_path in clips:
                meta = {
                    "filename":  clip_path.name,
                    "class_dir": class_dir.name,
                    **_video_meta(clip_path),
                }
                self._samples.append((clip_path, label, meta))

        logger.info(
            "VideoFolderLoader: %d clips across %d classes.",
            len(self._samples),
            len(self._class_names),
        )

    # ------------------------------------------------------------------
    # BaseDatasetLoader interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[tuple[Path, Optional[str], dict]]:
        for video_path, label, meta in self._samples:
            yield video_path, label, meta

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