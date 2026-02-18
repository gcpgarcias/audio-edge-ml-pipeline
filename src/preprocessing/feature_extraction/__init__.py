"""
Feature extraction sub-package.

Importing this package makes all built-in extractors available via the
registry (``get("audio_classical")`` etc.) and re-exports the concrete
classes for direct import.
"""

from .base import BaseFeatureExtractor, BaseDatasetLoader, FeatureSet
from .registry import get, list_extractors, register

# Audio extractors
from .audio.classical import AudioClassicalExtractor
from .audio.deep import AudioCQT, AudioMelSpectrogram, AudioMFCCSequence, AudioWaveform

# Image extractors
from .image.classical import ImageClassicalExtractor
from .image.deep import ImageMobileNetV2, ImagePixels

__all__ = [
    # Abstractions
    "BaseFeatureExtractor",
    "BaseDatasetLoader",
    "FeatureSet",
    "register",
    "get",
    "list_extractors",
    # Audio
    "AudioClassicalExtractor",
    "AudioMelSpectrogram",
    "AudioWaveform",
    "AudioCQT",
    "AudioMFCCSequence",
    # Image
    "ImageClassicalExtractor",
    "ImagePixels",
    "ImageMobileNetV2",
]