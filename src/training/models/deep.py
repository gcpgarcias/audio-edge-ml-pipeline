"""
Deep learning (Keras / TensorFlow) model trainers.

Registered names
----------------
``mlp``         Fully-connected Dense network — any flat ``(D,)`` input
``cnn``         Conv2D blocks + GlobalAveragePooling — ``(H, W[, C])`` input
``rnn``         Bidirectional LSTM — ``(T, D)`` sequence input
``transformer`` Multi-head self-attention blocks — ``(T, D)`` sequence input

Input normalisation
-------------------
All trainers accept raw float32 feature arrays.  Inputs are normalised to
zero mean / unit variance inside the model using a ``Normalization`` layer
fitted on the training set.

Shape handling
--------------
- Flat ``(D,)`` inputs fed to ``rnn`` or ``transformer`` are reshaped to
  ``(D, 1)`` — treating each feature as a 1-step time series.
- ``(H, W)`` inputs fed to ``cnn`` are expanded to ``(H, W, 1)``.

Persistence
-----------
Models are saved with ``model.save(path)`` (Keras SavedModel format) and
reloaded with ``tf.keras.models.load_model(path)``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .base import BaseTrainer, TrainResult
from . import register_model
from ..evaluate import (
    compute_metrics,
    save_classification_report,
    save_confusion_matrix_png,
    save_model_info,
    log_run_to_mlflow,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared Keras training logic
# ---------------------------------------------------------------------------

class KerasTrainer(BaseTrainer):
    """Generic wrapper around a compiled Keras model.

    Subclasses must implement :meth:`_build_model` which receives the
    (normalised) ``input_shape`` and ``n_classes`` and returns a compiled
    ``tf.keras.Model``.
    """

    model_type = "deep"

    def __init__(
        self,
        epochs:      int   = 50,
        batch_size:  int   = 32,
        dropout:     float = 0.3,
        learning_rate: float = 1e-3,
        **kwargs,
    ):
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.dropout       = dropout
        self.learning_rate = learning_rate
        self._extra        = kwargs
        self._model        = None

    # ------------------------------------------------------------------
    # Subclass hook
    # ------------------------------------------------------------------

    def _build_model(self, input_shape: tuple, n_classes: int):
        """Return a compiled Keras model.  Must be implemented by subclasses."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Input pre-processing
    # ------------------------------------------------------------------

    def _prepare_input(self, X: np.ndarray) -> np.ndarray:
        """Shape-coerce X to the expected ndim for this architecture."""
        return X  # overridden by CNN / RNN / Transformer subclasses

    # ------------------------------------------------------------------
    # BaseTrainer interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train:         np.ndarray,
        y_train:         np.ndarray,
        X_val:           np.ndarray,
        y_val:           np.ndarray,
        label_names:     list[str],
        run_name:        str,
        output_dir:      Path,
        mlflow_run,
        extra_callbacks: list = None,
    ) -> TrainResult:
        import tensorflow as tf

        X_train = self._prepare_input(X_train).astype(np.float32)
        X_val   = self._prepare_input(X_val).astype(np.float32)
        n_classes = len(label_names)

        self._model = self._build_model(X_train.shape[1:], n_classes)

        # Adapt the Normalization layer (always layer index 1: Input → Norm → ...)
        norm_layer = self._model.layers[1]
        if hasattr(norm_layer, "adapt"):
            norm_layer.adapt(X_train)

        # Build MLflow epoch callback
        class _MLflowCb(tf.keras.callbacks.Callback):
            def __init__(self, run):
                super().__init__()
                self._run = run
            def on_epoch_end(self, epoch, logs=None):
                if self._run is None or logs is None:
                    return
                import mlflow
                for k, v in logs.items():
                    mlflow.log_metric(k, float(v), step=epoch)

        # Per-epoch progress logger
        _total_epochs = self.epochs
        _model_name   = self.name

        class _ProgressCb(tf.keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self._prev_lr    = None
                self._last_epoch = 0

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                self._last_epoch = epoch + 1
                loss     = logs.get("loss",         float("nan"))
                acc      = logs.get("accuracy",     float("nan"))
                val_loss = logs.get("val_loss",     float("nan"))
                val_acc  = logs.get("val_accuracy", float("nan"))

                try:
                    current_lr = float(self.model.optimizer.learning_rate)
                except Exception:
                    current_lr = None

                lr_tag = ""
                if current_lr is not None:
                    if self._prev_lr is not None and current_lr < self._prev_lr - 1e-9:
                        lr_tag = f"  lr={current_lr:.2e}↓"
                    self._prev_lr = current_lr

                logger.info(
                    "[%s] Epoch %3d/%d  loss=%.4f  acc=%.4f  val_loss=%.4f  val_acc=%.4f%s",
                    _model_name, self._last_epoch, _total_epochs,
                    loss, acc, val_loss, val_acc, lr_tag,
                )

            def on_train_end(self, logs=None):
                if self._last_epoch < _total_epochs:
                    logger.info(
                        "[%s] Early stopped at epoch %d/%d",
                        _model_name, self._last_epoch, _total_epochs,
                    )

        callbacks = [
            _MLflowCb(mlflow_run),
            _ProgressCb(),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
            ),
            *(extra_callbacks or []),
        ]

        logger.info("Training %s on %d samples ...", self.name, len(X_train))
        self._model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0,
        )

        # Evaluate
        y_pred_val = np.argmax(self._model.predict(X_val, verbose=0), axis=1)
        val_metrics = compute_metrics(y_val, y_pred_val, label_names=label_names)

        # Save model
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "model.keras"
        self._model.save(model_path)
        model_size_kb = sum(
            f.stat().st_size for f in model_path.rglob("*") if f.is_file()
        ) / 1024 if model_path.is_dir() else model_path.stat().st_size / 1024

        # Build params dict (needed by save_model_info and log_run_to_mlflow)
        params = {
            "model":         self.name,
            "epochs":        self.epochs,
            "batch_size":    self.batch_size,
            "dropout":       self.dropout,
            "learning_rate": self.learning_rate,
        }
        params.update({k: str(v) for k, v in self._extra.items()})

        # Write per-run artefacts via evaluate helpers
        save_classification_report(y_val, y_pred_val, label_names,
                                   output_dir / "classification_report.txt")
        save_confusion_matrix_png(val_metrics.get("confusion_matrix", []),
                                  label_names, output_dir / "confusion_matrix.png")
        save_model_info(output_dir, self.name, run_name, val_metrics, params, model_size_kb)

        # Log to MLflow
        val_metrics["model_size_kb"] = model_size_kb
        log_run_to_mlflow(mlflow_run, params, val_metrics, output_dir)
        if mlflow_run is not None:
            import mlflow
            mlflow.log_artifact(str(model_path) if model_path.is_file()
                                else str(output_dir / "model.keras"))

        return TrainResult(
            model_name=self.name,
            run_id=mlflow_run.info.run_id if mlflow_run else "",
            output_dir=output_dir,
            metrics=val_metrics,
            model_size_kb=model_size_kb,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._prepare_input(X).astype(np.float32)
        return np.argmax(self._model.predict(X, verbose=0), axis=1)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        X = self._prepare_input(X).astype(np.float32)
        return self._model.predict(X, verbose=0)

    def save(self, path: Path) -> None:
        self._model.save(path)

    @classmethod
    def load(cls, path: Path) -> "KerasTrainer":
        import tensorflow as tf
        inst = cls.__new__(cls)
        inst._model = tf.keras.models.load_model(path)
        return inst


# ---------------------------------------------------------------------------
# MLP — fully-connected Dense network
# ---------------------------------------------------------------------------

@register_model
class MLPTrainer(KerasTrainer):
    """Multi-Layer Perceptron.

    Suited for flat feature vectors: ``audio_classical``, ``image_mobilenet_v2``,
    ``tabular_classical``, etc.

    Parameters
    ----------
    hidden_units:   List of hidden layer sizes (default ``[256, 128]``).
    dropout:        Dropout rate applied after each hidden layer.
    epochs:         Maximum training epochs (EarlyStopping may stop earlier).
    batch_size:     Mini-batch size.
    learning_rate:  Initial Adam learning rate.
    """

    name       = "mlp"
    model_type = "deep"

    def __init__(
        self,
        hidden_units: list[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_units = hidden_units or [256, 128]

    def _build_model(self, input_shape: tuple, n_classes: int):
        import tensorflow as tf

        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Normalization()(inputs)
        for units in self.hidden_units:
            x = tf.keras.layers.Dense(units, activation="relu")(x)
            x = tf.keras.layers.Dropout(self.dropout)(x)
        outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    @classmethod
    def load(cls, path: Path) -> "MLPTrainer":
        import tensorflow as tf
        inst = cls.__new__(cls)
        inst._model = tf.keras.models.load_model(path)
        return inst


# ---------------------------------------------------------------------------
# CNN — 2-D convolutional network
# ---------------------------------------------------------------------------

@register_model
class CNNTrainer(KerasTrainer):
    """2-D Convolutional Neural Network.

    Suited for image-like 2-D features: spectrograms (``audio_mel_spec``,
    ``audio_mfcc_seq``), raw image pixels, etc.

    Inputs with shape ``(H, W)`` are automatically expanded to ``(H, W, 1)``.

    Parameters
    ----------
    filters:        List of filter counts per Conv block (default ``[32, 64]``).
    dropout:        Dropout rate.
    epochs, batch_size, learning_rate: standard training hyperparameters.
    """

    name       = "cnn"
    model_type = "deep"

    def __init__(self, filters=None, n_blocks: int = None, **kwargs):
        super().__init__(**kwargs)
        if filters is None:
            filters = [32, 64]
        if isinstance(filters, int):
            filters = [filters] * (n_blocks or 2)
        self.filters = list(filters)

    def _prepare_input(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 2:          # (N, D) flat — shouldn't happen but guard
            return X[:, :, np.newaxis]
        if X.ndim == 3:          # (N, H, W) → (N, H, W, 1)
            return X[:, :, :, np.newaxis]
        return X                 # (N, H, W, C) — pass through

    def _build_model(self, input_shape: tuple, n_classes: int):
        import tensorflow as tf

        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Normalization()(inputs)
        for f in self.filters:
            x = tf.keras.layers.Conv2D(f, (3, 3), activation="relu", padding="same")(x)
            x = tf.keras.layers.MaxPooling2D((2, 2))(x)
            x = tf.keras.layers.Dropout(self.dropout)(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)
        outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    @classmethod
    def load(cls, path: Path) -> "CNNTrainer":
        import tensorflow as tf
        inst = cls.__new__(cls)
        inst._model = tf.keras.models.load_model(path)
        return inst


# ---------------------------------------------------------------------------
# RNN — Bidirectional LSTM
# ---------------------------------------------------------------------------

@register_model
class RNNTrainer(KerasTrainer):
    """Bidirectional LSTM sequence model.

    Suited for time-series feature inputs: ``audio_mfcc_seq`` ``(n_mfcc, T)``,
    ``video_mobilenet_v2_seq`` ``(T, 1280)``, etc.

    Flat ``(D,)`` inputs are reshaped to ``(D, 1)``.

    Parameters
    ----------
    units:      LSTM hidden units (default 128).
    n_layers:   Number of stacked Bi-LSTM layers (default 1).
    dropout, epochs, batch_size, learning_rate: standard hyperparameters.
    """

    name       = "rnn"
    model_type = "deep"

    def __init__(self, units: int = 128, n_layers: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.units    = units
        self.n_layers = n_layers

    def _prepare_input(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 2:          # (N, D) → (N, D, 1)
            return X[:, :, np.newaxis]
        return X                 # already (N, T, D)

    def _build_model(self, input_shape: tuple, n_classes: int):
        import tensorflow as tf

        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Normalization()(inputs)
        for i in range(self.n_layers):
            return_seq = i < self.n_layers - 1
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(self.units, return_sequences=return_seq,
                                     dropout=self.dropout)
            )(x)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)
        outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    @classmethod
    def load(cls, path: Path) -> "RNNTrainer":
        import tensorflow as tf
        inst = cls.__new__(cls)
        inst._model = tf.keras.models.load_model(path)
        return inst


# ---------------------------------------------------------------------------
# Transformer — Multi-head self-attention
# ---------------------------------------------------------------------------

@register_model
class TransformerTrainer(KerasTrainer):
    """Simple 1-D Transformer encoder (multi-head self-attention).

    Suited for sequence inputs: ``audio_mfcc_seq``, ``video_mobilenet_v2_seq``.

    Flat ``(D,)`` inputs are reshaped to ``(D, 1)``.

    Parameters
    ----------
    num_heads:      Number of attention heads (default 4).
    ff_dim:         Feed-forward hidden dimension (default 128).
    n_blocks:       Number of Transformer encoder blocks (default 2).
    dropout, epochs, batch_size, learning_rate: standard hyperparameters.
    """

    name       = "transformer"
    model_type = "deep"

    def __init__(self, num_heads: int = 4, ff_dim: int = 128, n_blocks: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.ff_dim    = ff_dim
        self.n_blocks  = n_blocks

    def _prepare_input(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 2:          # (N, D) → (N, D, 1)
            return X[:, :, np.newaxis]
        return X                 # already (N, T, D)

    def _build_model(self, input_shape: tuple, n_classes: int):
        import tensorflow as tf

        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Normalization()(inputs)

        for _ in range(self.n_blocks):
            # Multi-head self-attention
            attn = tf.keras.layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=max(1, input_shape[-1] // self.num_heads)
            )(x, x)
            attn = tf.keras.layers.Dropout(self.dropout)(attn)
            x    = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn)

            # Feed-forward
            ff = tf.keras.layers.Dense(self.ff_dim, activation="relu")(x)
            ff = tf.keras.layers.Dense(input_shape[-1])(ff)
            ff = tf.keras.layers.Dropout(self.dropout)(ff)
            x  = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ff)

        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)
        outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    @classmethod
    def load(cls, path: Path) -> "TransformerTrainer":
        import tensorflow as tf
        inst = cls.__new__(cls)
        inst._model = tf.keras.models.load_model(path)
        return inst


