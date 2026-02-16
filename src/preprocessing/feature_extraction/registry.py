"""
Feature extractor registry.

Extractors self-register via the ``@register`` class decorator.  The registry
maps string names to extractor *classes* (not instances), so callers control
instantiation and can pass constructor arguments.

Usage
-----
Registering::

    from src.preprocessing.feature_extraction.registry import register
    from src.preprocessing.feature_extraction.base import BaseFeatureExtractor

    @register
    class MyExtractor(BaseFeatureExtractor):
        name         = "my_extractor"
        feature_type = "classical"
        modality     = "audio"
        ...

Resolving::

    from src.preprocessing.feature_extraction.registry import get, list_extractors

    ExtractorClass = get("audio_classical")
    extractor = ExtractorClass(n_mfcc=40, sample_rate=22050)

    print(list_extractors())
    # ['audio_classical', 'audio_cqt', 'audio_mel_spectrogram', 'audio_waveform']
"""

from __future__ import annotations

from typing import Type

from .base import BaseFeatureExtractor

_REGISTRY: dict[str, Type[BaseFeatureExtractor]] = {}


def register(cls: Type[BaseFeatureExtractor]) -> Type[BaseFeatureExtractor]:
    """Class decorator that registers *cls* under ``cls.name``.

    The decorated class is returned unchanged, so it can still be used
    directly by importing the defining module.

    Raises
    ------
    TypeError
        If *cls* does not define a ``name`` class attribute.
    ValueError
        If another extractor is already registered under the same name.
    """
    if not hasattr(cls, "name") or not isinstance(cls.name, str):
        raise TypeError(
            f"{cls.__qualname__} must define a string class attribute 'name'."
        )
    if cls.name in _REGISTRY:
        raise ValueError(
            f"An extractor named '{cls.name}' is already registered "
            f"({_REGISTRY[cls.name].__qualname__}). "
            "Use a unique name or remove the duplicate."
        )
    _REGISTRY[cls.name] = cls
    return cls


def get(name: str) -> Type[BaseFeatureExtractor]:
    """Look up an extractor class by name.

    Raises
    ------
    KeyError
        If no extractor is registered under *name*.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"No extractor named '{name!r}'. "
            f"Available extractors: {list_extractors()}"
        )
    return _REGISTRY[name]


def list_extractors() -> list[str]:
    """Return a sorted list of all registered extractor names."""
    return sorted(_REGISTRY)