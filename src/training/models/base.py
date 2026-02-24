"""
Base abstractions for the model training layer.

Provides:
- TrainResult: uniform result container returned by every trainer.
- BaseTrainer: abstract base class all trainers must implement.

The interface is intentionally minimal so that both sklearn estimators and
Keras models can be wrapped behind the same API.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """Uniform result container returned by :meth:`BaseTrainer.fit`.

    Parameters
    ----------
    model_name:
        Registry key of the trained model (e.g. ``"svm"``).
    run_id:
        MLflow run ID.  Empty string when MLflow tracking is disabled.
    output_dir:
        Directory where the model artefacts were saved.
    metrics:
        Evaluation metrics on the validation set, and optionally on a
        held-out test set.  At minimum contains ``val_accuracy`` and
        ``val_f1_macro``.
    model_size_kb:
        Size of the serialised model file(s) in kilobytes.
    params:
        Hyperparameters passed to the trainer constructor.
    """

    model_name:    str
    run_id:        str
    output_dir:    Path
    metrics:       dict
    model_size_kb: float
    params:        dict = field(default_factory=dict)

    def __repr__(self) -> str:
        acc = self.metrics.get("val_accuracy", float("nan"))
        return (
            f"TrainResult(model={self.model_name!r}, "
            f"val_accuracy={acc:.4f}, "
            f"size={self.model_size_kb:.1f} KB, "
            f"output={self.output_dir})"
        )


class BaseTrainer(ABC):
    """Abstract base class for all model trainers.

    Subclasses must declare two class-level attributes::

        name:       str   # registry key — globally unique
        model_type: str   # "classical" | "deep"

    and implement :meth:`fit`, :meth:`predict`, :meth:`save`, and
    :meth:`load`.

    The :meth:`predict_proba` method is optional; it should be overridden
    whenever the underlying model supports soft probability outputs (needed
    for ROC-AUC, calibrated probability displays, etc.).
    """

    name:       str   # set by concrete subclass
    model_type: str   # "classical" | "deep"

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(
        self,
        X_train:     np.ndarray,
        y_train:     np.ndarray,
        X_val:       np.ndarray,
        y_val:       np.ndarray,
        label_names: list[str],
        run_name:    str,
        output_dir:  Path,
        mlflow_run,                # mlflow.ActiveRun or None
    ) -> TrainResult:
        """Train the model and return a :class:`TrainResult`.

        Parameters
        ----------
        X_train, y_train:
            Training features and integer class labels.
        X_val, y_val:
            Validation features and integer class labels.
        label_names:
            Ordered list mapping integer index → class name string.
        run_name:
            Human-readable name for this training run.
        output_dir:
            Directory for model artefacts (created by the caller).
        mlflow_run:
            Active MLflow run context, or *None* when MLflow is disabled.
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return integer class predictions for *X*."""
        ...

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Return class probability estimates for *X*, shape ``(N, C)``.

        Returns *None* by default; override when soft outputs are available.
        """
        return None

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the trained model to *path*.

        *path* should be a file path (not a directory).  The file extension
        is determined by the subclass (e.g. ``.joblib`` or ``.keras``).
        """
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseTrainer":
        """Reload a previously saved trainer from *path*."""
        ...

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def flatten(X: np.ndarray) -> np.ndarray:
        """Flatten features to 2-D ``(N, D)`` if ndim > 2.

        Deep features such as ``(T, H, W, C)`` from video or ``(T, 1280)``
        from MobileNetV2 sequences need to be flattened before classical
        estimators can consume them.
        """
        if X.ndim > 2:
            return X.reshape(X.shape[0], -1)
        return X
