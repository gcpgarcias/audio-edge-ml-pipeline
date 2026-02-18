"""
Dataset loader sub-package.

Each loader yields ``(sample_path, label, metadata)`` triples and is
compatible with any :class:`~src.preprocessing.feature_extraction.base.BaseFeatureExtractor`.
"""

from .birdeep_loader import BIRDeepImageLoader, BIRDeepLoader
from .image_folder_loader import ImageFolderLoader

__all__ = [
    "BIRDeepLoader",
    "BIRDeepImageLoader",
    "ImageFolderLoader",
]