"""
Dataset loader sub-package.

Each loader yields ``(sample_path, label, metadata)`` triples and is
compatible with any :class:`~src.preprocessing.feature_extraction.base.BaseFeatureExtractor`.
``sample_path`` is *None* for in-memory loaders (tabular, JSON/CSV text).
"""

from .audio_folder_loader import AudioFolderLoader
from .birdeep_loader import BIRDeepImageLoader, BIRDeepLoader
from .image_folder_loader import ImageFolderLoader
from .tabular_loader import TabularLoader
from .text_loader import TextCSVLoader, TextFolderLoader, TextJSONLoader
from .video_folder_loader import VideoFolderLoader

__all__ = [
    # Audio
    "AudioFolderLoader",
    "BIRDeepLoader",
    # Image
    "BIRDeepImageLoader",
    "ImageFolderLoader",
    # Text
    "TextFolderLoader",
    "TextJSONLoader",
    "TextCSVLoader",
    # Tabular
    "TabularLoader",
    # Video
    "VideoFolderLoader",
]