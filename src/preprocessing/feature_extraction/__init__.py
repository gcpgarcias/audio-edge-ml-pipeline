"""
Feature extraction sub-package.

Importing this package triggers registration of all built-in extractors so
that ``registry.get("audio_classical")`` and friends work without explicit
imports of the extractor modules.
"""

from .base import BaseFeatureExtractor, BaseDatasetLoader, FeatureSet
from .registry import get, list_extractors, register

# --- Auto-import extractor modules so their @register decorators fire ---
from .audio import classical as _audio_classical  # noqa: F401
from .audio import deep as _audio_deep            # noqa: F401

__all__ = [
    "BaseFeatureExtractor",
    "BaseDatasetLoader",
    "FeatureSet",
    "register",
    "get",
    "list_extractors",
]