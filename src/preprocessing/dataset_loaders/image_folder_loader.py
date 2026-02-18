"""
Generic image folder dataset loader.

Handles the standard ``class-per-subfolder`` layout that most image
classification benchmarks use:

    <dataset_root>/
        class_a/
            image_001.jpg
            image_002.png
        class_b/
            image_001.jpeg
        …

Optionally a split subdirectory (``train``, ``test``, ``val``) can sit
between the root and the class directories:

    <dataset_root>/
        train/
            class_a/ …
            class_b/ …
        test/
            class_a/ …

Usage
-----
::

    from src.preprocessing.dataset_loaders.image_folder_loader import ImageFolderLoader
    from src.preprocessing.feature_extraction import get

    loader    = ImageFolderLoader("path/to/dataset")
    extractor = get("image_classical")()
    fs = extractor.extract_dataset(loader)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

from ..feature_extraction.base import BaseDatasetLoader

logger = logging.getLogger(__name__)

# Image file extensions recognised by the loader (case-insensitive).
_IMAGE_SUFFIXES: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)


class ImageFolderLoader(BaseDatasetLoader):
    """Load images from a class-per-subfolder directory tree.

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
        dot).  Defaults to common raster image formats.
    class_names:
        If given, only images belonging to these class folders are loaded,
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
            else _IMAGE_SUFFIXES
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

        # Build flat list of (image_path, class_label)
        self._samples: list[tuple[Path, str]] = []
        for class_dir, label in zip(class_dirs, self._class_names):
            if not class_dir.is_dir():
                logger.warning("Class directory not found: %s (skipping)", class_dir)
                continue
            imgs = [
                p for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in self._extensions
            ]
            if not imgs:
                logger.warning("No images found in: %s", class_dir)
            for img_path in sorted(imgs):
                self._samples.append((img_path, label))

        logger.info(
            "ImageFolderLoader: %d images across %d classes.",
            len(self._samples),
            len(self._class_names),
        )

    # ------------------------------------------------------------------
    # BaseDatasetLoader interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[tuple[Path, Optional[str], dict]]:
        for img_path, label in self._samples:
            meta = {"filename": img_path.name, "class_dir": img_path.parent.name}
            yield img_path, label, meta

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