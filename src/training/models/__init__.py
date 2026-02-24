"""
Model trainer registry for Stage 3.

Usage
-----
from src.training.models import get_model, list_models, register_model

# Instantiate by name (kwargs forwarded to trainer __init__)
trainer = get_model("svm")(C=10.0)
trainer = get_model("cnn")(filters=32, n_blocks=2)

# List all registered names
print(list_models())
# ['cnn', 'decision_tree', 'knn', 'kmeans', 'lda', 'mlp',
#  'pca_svm', 'random_forest', 'rnn', 'svm', 'transformer']

Registration
------------
Decorating a :class:`~src.training.models.base.BaseTrainer` subclass with
``@register_model`` adds it to the global registry under ``cls.name``.
Both classical and deep trainer modules apply this decorator at import time,
so importing *this* package is sufficient to populate the registry.
"""

from __future__ import annotations

import logging
from typing import Type

from .base import BaseTrainer, TrainResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Type[BaseTrainer]] = {}


def register_model(cls: Type[BaseTrainer]) -> Type[BaseTrainer]:
    """Class decorator — register *cls* under ``cls.name``.

    Raises
    ------
    TypeError
        If *cls* does not subclass :class:`BaseTrainer`.
    ValueError
        If a trainer with the same name is already registered.
    AttributeError
        If *cls* does not define a ``name`` class attribute.
    """
    if not issubclass(cls, BaseTrainer):
        raise TypeError(f"@register_model expects a BaseTrainer subclass, got {cls!r}")
    if not hasattr(cls, "name") or not isinstance(cls.name, str):
        raise AttributeError(f"{cls!r} must define a 'name' class attribute (str)")
    if cls.name in _REGISTRY:
        # Allow re-registration of the same class (e.g. module reload in notebooks)
        if _REGISTRY[cls.name] is not cls:
            raise ValueError(
                f"Trainer name '{cls.name}' is already registered by "
                f"{_REGISTRY[cls.name]!r}. Choose a different name."
            )
        return cls
    _REGISTRY[cls.name] = cls
    logger.debug("Registered model trainer: %s (%s)", cls.name, cls.__name__)
    return cls


def get_model(name: str) -> Type[BaseTrainer]:
    """Return the trainer class registered under *name*.

    Parameters
    ----------
    name:
        Registry key (e.g. ``"svm"``, ``"cnn"``).

    Raises
    ------
    KeyError
        If no trainer is registered under *name*.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(
            f"No trainer registered under '{name}'. "
            f"Available: {available or '(none — did you import the trainer modules?)'}"
        )
    return _REGISTRY[name]


def list_models() -> list[str]:
    """Return a sorted list of all registered trainer names."""
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# Trigger registration by importing all concrete trainer modules
# ---------------------------------------------------------------------------

from .classical import (  # noqa: E402, F401
    SVMTrainer,
    LDATrainer,
    DecisionTreeTrainer,
    RandomForestTrainer,
    KNNTrainer,
    KMeansTrainer,
    PCASVMTrainer,
)

from .deep import (  # noqa: E402, F401
    MLPTrainer,
    CNNTrainer,
    RNNTrainer,
    TransformerTrainer,
)

__all__ = [
    # Registry API
    "register_model",
    "get_model",
    "list_models",
    # Base types (re-exported for convenience)
    "BaseTrainer",
    "TrainResult",
    # Classical trainers
    "SVMTrainer",
    "LDATrainer",
    "DecisionTreeTrainer",
    "RandomForestTrainer",
    "KNNTrainer",
    "KMeansTrainer",
    "PCASVMTrainer",
    # Deep trainers
    "MLPTrainer",
    "CNNTrainer",
    "RNNTrainer",
    "TransformerTrainer",
]