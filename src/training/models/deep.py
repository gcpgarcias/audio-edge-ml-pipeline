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

# Default temperature and alpha for knowledge distillation
_KD_TEMPERATURE = 4.0
_KD_ALPHA       = 0.7


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

    def _architecture_params(self) -> dict:
        """Subclass-specific hyperparams to include in MLflow logging.

        Override in each subclass to expose architecture parameters
        (e.g. hidden_units, filters) that are not in the base KerasTrainer.
        """
        return {}

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

        # Optional: load pretrained weights (fine-tuning).
        # Copies weights layer-by-layer by name, skipping the Normalization layer
        # so that the norm statistics stay fitted to the current training data.
        pretrained_path = self._extra.pop("pretrained_model", None)
        if pretrained_path:
            logger.info("Loading pretrained weights from %s", pretrained_path)
            src = tf.keras.models.load_model(pretrained_path)
            transferred, skipped = 0, 0
            for layer in self._model.layers:
                if isinstance(layer, tf.keras.layers.Normalization):
                    skipped += 1
                    continue
                try:
                    src_layer = src.get_layer(layer.name)
                    layer.set_weights(src_layer.get_weights())
                    transferred += 1
                except (ValueError, AttributeError):
                    skipped += 1
            logger.info("Pretrained weights: %d layers transferred, %d skipped", transferred, skipped)

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
        params.update({k: str(v) for k, v in self._architecture_params().items()})
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

    def _architecture_params(self) -> dict:
        return {"hidden_units": self.hidden_units}

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

    def __init__(self, filters=None, n_blocks: int = None,
                 first_stride: int = 1, second_stride: int = 1, **kwargs):
        super().__init__(**kwargs)
        if filters is None:
            filters = [32, 64]
        if isinstance(filters, int):
            filters = [filters] * (n_blocks or 2)
        self.filters       = list(filters)
        self.first_stride  = first_stride
        self.second_stride = second_stride

    def _architecture_params(self) -> dict:
        return {"filters": self.filters, "first_stride": self.first_stride,
                "second_stride": self.second_stride}

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
        for i, f in enumerate(self.filters):
            if i == 0:
                stride = (self.first_stride, self.first_stride)
            elif i == 1:
                stride = (self.second_stride, self.second_stride)
            else:
                stride = (1, 1)
            x = tf.keras.layers.Conv2D(
                f, (3, 3), strides=stride, activation="relu", padding="same")(x)
            # Skip MaxPool for a block whose conv already strides by 2
            use_stride = (i == 0 and self.first_stride > 1) or \
                         (i == 1 and self.second_stride > 1)
            if not use_stride:
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

    def _architecture_params(self) -> dict:
        return {"units": self.units, "n_layers": self.n_layers}

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

    def _architecture_params(self) -> dict:
        return {"num_heads": self.num_heads, "ff_dim": self.ff_dim, "n_blocks": self.n_blocks}

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


# ---------------------------------------------------------------------------
# Serialisable preprocessing layer for EfficientNetTeacherTrainer
# ---------------------------------------------------------------------------

def _make_prep_layer(target_h: int, target_w: int):
    """Return a serialisable Keras layer that stacks mono → RGB and resizes."""
    import tensorflow as tf

    class PrepRGBResize(tf.keras.layers.Layer):
        def __init__(self, target_h: int, target_w: int, **kwargs):
            super().__init__(**kwargs)
            self.target_h = target_h
            self.target_w = target_w

        def call(self, x):
            if x.shape[-1] != 3:
                x = tf.repeat(x, 3, axis=-1)
            return tf.image.resize(x, (self.target_h, self.target_w))

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"target_h": self.target_h, "target_w": self.target_w})
            return cfg

    return PrepRGBResize(target_h, target_w, name="prep_rgb_resize")


# ---------------------------------------------------------------------------
# EfficientNetTeacherTrainer — fine-tuned EfficientNet-B0 teacher
# ---------------------------------------------------------------------------

@register_model
class EfficientNetTeacherTrainer(KerasTrainer):
    """EfficientNet-B0 fine-tuned on mel spectrograms — intended as a
    distillation teacher for :class:`DistillationCNNTrainer`.

    Training runs in two phases:

    1. **Frozen backbone** (``warmup_epochs``): only the classification head
       trains; ImageNet weights are preserved.
    2. **Partial unfreeze** (remaining epochs): the top ``unfreeze_layers``
       layers of the backbone are unfrozen and trained with a lower LR
       (``learning_rate * fine_tune_lr_factor``).

    Input handling
    --------------
    Accepts ``(H, W)`` or ``(H, W, 1)`` mono spectrograms.  The channel is
    stacked ×3 to match EfficientNet's RGB expectation, then bilinearly
    resized to ``(target_h, target_w)`` (default 128×128).

    Parameters
    ----------
    warmup_epochs:        Epochs with frozen backbone (default 10).
    unfreeze_layers:      Number of top backbone layers to unfreeze in phase 2
                          (default 20).
    fine_tune_lr_factor:  LR multiplier for the fine-tune phase (default 0.1).
    target_h, target_w:   Resize target — must be ≥ 32 (default 128×128).
    dropout:              Dropout before the classification head (default 0.3).
    epochs, batch_size, learning_rate: standard hyperparameters.
    """

    name       = "efficientnet_teacher"
    model_type = "deep"

    def __init__(
        self,
        warmup_epochs:       int   = 10,
        unfreeze_layers:     int   = 20,
        fine_tune_lr_factor: float = 0.1,
        target_h:            int   = 128,
        target_w:            int   = 128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.warmup_epochs       = warmup_epochs
        self.unfreeze_layers     = unfreeze_layers
        self.fine_tune_lr_factor = fine_tune_lr_factor
        self.target_h            = target_h
        self.target_w            = target_w

    def _architecture_params(self) -> dict:
        return {
            "warmup_epochs":       self.warmup_epochs,
            "unfreeze_layers":     self.unfreeze_layers,
            "fine_tune_lr_factor": self.fine_tune_lr_factor,
            "target_h":            self.target_h,
            "target_w":            self.target_w,
        }

    def _prepare_input(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:          # (N, H, W) → (N, H, W, 1)
            return X[:, :, :, np.newaxis]
        return X                 # (N, H, W, C)

    def _build_model(self, input_shape: tuple, n_classes: int):
        import tensorflow as tf

        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Normalization()(inputs)

        # Mono → RGB + resize (serialisable custom layer — no Lambda closure)
        x = _make_prep_layer(self.target_h, self.target_w)(x)

        # EfficientNet-B0 backbone (frozen initially)
        # macOS ships without root CA certs for the Python SSL bundle; patch
        # the context so the Keras weights download can proceed.
        import ssl
        _orig_ctx = ssl._create_default_https_context
        ssl._create_default_https_context = ssl._create_unverified_context
        try:
            backbone = tf.keras.applications.EfficientNetB0(
                include_top=False,
                weights="imagenet",
                input_shape=(self.target_h, self.target_w, 3),
                pooling="avg",
            )
        finally:
            ssl._create_default_https_context = _orig_ctx
        backbone.trainable = False

        x = backbone(x, training=False)
        x = tf.keras.layers.Dropout(self.dropout)(x)
        outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(
        self,
        X_train:     np.ndarray,
        y_train:     np.ndarray,
        X_val:       np.ndarray,
        y_val:       np.ndarray,
        label_names: list[str],
        run_name:    str,
        output_dir:  Path,
        mlflow_run,
        extra_callbacks: list = None,
    ) -> TrainResult:
        import tensorflow as tf

        X_train = self._prepare_input(X_train).astype(np.float32)
        X_val   = self._prepare_input(X_val).astype(np.float32)
        n_classes = len(label_names)

        self._model = self._build_model(X_train.shape[1:], n_classes)
        norm_layer  = self._model.layers[1]
        if hasattr(norm_layer, "adapt"):
            norm_layer.adapt(X_train)

        # ── Shared callbacks ──────────────────────────────────────────
        class _MLflowCb(tf.keras.callbacks.Callback):
            def __init__(self, run, offset=0):
                super().__init__()
                self._run    = run
                self._offset = offset
            def on_epoch_end(self, epoch, logs=None):
                if self._run is None or logs is None:
                    return
                import mlflow
                for k, v in logs.items():
                    mlflow.log_metric(k, float(v), step=self._offset + epoch)

        _name = self.name

        class _ProgressCb(tf.keras.callbacks.Callback):
            def __init__(self, total, offset=0, phase=""):
                super().__init__()
                self._total  = total
                self._offset = offset
                self._phase  = phase
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                logger.info(
                    "[%s%s] Epoch %3d/%d  loss=%.4f  acc=%.4f  val_loss=%.4f  val_acc=%.4f",
                    _name, self._phase,
                    self._offset + epoch + 1, self._total,
                    logs.get("loss", float("nan")),
                    logs.get("accuracy", float("nan")),
                    logs.get("val_loss", float("nan")),
                    logs.get("val_accuracy", float("nan")),
                )

        # ── Phase 1: frozen backbone ──────────────────────────────────
        logger.info("[%s] Phase 1 — warmup %d epochs (backbone frozen)",
                    self.name, self.warmup_epochs)
        self._model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.warmup_epochs,
            batch_size=self.batch_size,
            callbacks=[
                _MLflowCb(mlflow_run, offset=0),
                _ProgressCb(self.epochs, offset=0, phase="/warmup"),
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=5, restore_best_weights=True
                ),
            ],
            verbose=0,
        )

        # ── Phase 2: partial unfreeze ─────────────────────────────────
        fine_tune_epochs = self.epochs - self.warmup_epochs
        if fine_tune_epochs > 0:
            # Unfreeze top N layers of the backbone (layer index 3 = backbone)
            backbone = self._model.layers[3]
            backbone.trainable = True
            for layer in backbone.layers[:-self.unfreeze_layers]:
                layer.trainable = False

            logger.info(
                "[%s] Phase 2 — fine-tune %d epochs, top %d backbone layers unfrozen",
                self.name, fine_tune_epochs, self.unfreeze_layers,
            )
            self._model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    self.learning_rate * self.fine_tune_lr_factor
                ),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            self._model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=fine_tune_epochs,
                batch_size=self.batch_size,
                callbacks=[
                    _MLflowCb(mlflow_run, offset=self.warmup_epochs),
                    _ProgressCb(self.epochs, offset=self.warmup_epochs, phase="/fine-tune"),
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss", patience=10, restore_best_weights=True
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7
                    ),
                    *(extra_callbacks or []),
                ],
                verbose=0,
            )

        # ── Evaluate and save ─────────────────────────────────────────
        y_pred_val  = np.argmax(self._model.predict(X_val, verbose=0), axis=1)
        val_metrics = compute_metrics(y_val, y_pred_val, label_names=label_names)

        output_dir.mkdir(parents=True, exist_ok=True)
        model_path   = output_dir / "model.keras"
        self._model.save(model_path)
        model_size_kb = (
            sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file()) / 1024
            if model_path.is_dir() else model_path.stat().st_size / 1024
        )

        params = {
            "model":         self.name,
            "epochs":        self.epochs,
            "batch_size":    self.batch_size,
            "dropout":       self.dropout,
            "learning_rate": self.learning_rate,
        }
        params.update({k: str(v) for k, v in self._architecture_params().items()})

        save_classification_report(y_val, y_pred_val, label_names,
                                   output_dir / "classification_report.txt")
        save_confusion_matrix_png(val_metrics.get("confusion_matrix", []),
                                  label_names, output_dir / "confusion_matrix.png")
        save_model_info(output_dir, self.name, run_name, val_metrics, params, model_size_kb)

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

    @classmethod
    def load(cls, path: Path) -> "EfficientNetTeacherTrainer":
        import tensorflow as tf
        inst = cls.__new__(cls)
        inst._model = tf.keras.models.load_model(path)
        return inst


# ---------------------------------------------------------------------------
# DistillationCNNTrainer — tiny CNN student trained with soft teacher labels
# ---------------------------------------------------------------------------

@register_model
class DistillationCNNTrainer(KerasTrainer):
    """Tiny CNN student trained with knowledge distillation.

    A soft-label loss (KL divergence on temperature-scaled logits from a
    pre-trained teacher) is blended with the standard cross-entropy hard-label
    loss.  The student architecture mirrors :class:`CNNTrainer` but defaults
    to very few filters, keeping the resulting ``.tflite`` INT8 model under
    ~200 KB.

    Parameters
    ----------
    teacher_model_path:
        **Required.** Path to a saved Keras teacher model
        (any architecture that outputs ``(N, n_classes)`` probabilities).
    filters:
        List of filter counts per Conv2D block (default ``[16, 16, 16]``).
    temperature:
        Distillation temperature *T* — higher values soften the teacher
        distribution more (default ``4.0``).
    alpha:
        Weight of the soft-label loss; ``1 - alpha`` is the weight of the
        hard-label loss (default ``0.7``).
    dropout, epochs, batch_size, learning_rate: standard hyperparameters.

    Notes
    -----
    - Input must be a 2-D spectrogram ``(H, W)`` or ``(H, W, C)``.  Flat
      ``(D,)`` inputs are **not** supported.
    - The teacher is loaded once at the start of :meth:`fit` and used only
      in inference mode to generate soft targets; its weights are frozen.
    - Soft targets are pre-computed over the full training set before
      training begins, so there is no per-batch teacher forward pass.
    """

    name       = "distillation_cnn"
    model_type = "deep"

    def __init__(
        self,
        teacher_model_path: str,
        filters=None,
        temperature: float = _KD_TEMPERATURE,
        alpha:       float = _KD_ALPHA,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.teacher_model_path = teacher_model_path
        self.filters     = list(filters or [16, 16, 16])
        self.temperature = temperature
        self.alpha       = alpha

    def _architecture_params(self) -> dict:
        return {
            "filters":     self.filters,
            "temperature": self.temperature,
            "alpha":       self.alpha,
        }

    def _prepare_input(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:          # (N, H, W) → (N, H, W, 1)
            return X[:, :, :, np.newaxis]
        return X                 # (N, H, W, C) pass through

    def _build_model(self, input_shape: tuple, n_classes: int):
        import tensorflow as tf

        inputs = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Normalization()(inputs)
        for f in self.filters:
            x = tf.keras.layers.Conv2D(f, (3, 3), activation="relu", padding="same")(x)
            x = tf.keras.layers.MaxPooling2D((2, 2))(x)
            x = tf.keras.layers.Dropout(self.dropout)(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)
        # Linear output — distillation loss works on logits
        outputs = tf.keras.layers.Dense(n_classes)(x)

        model = tf.keras.Model(inputs, outputs)
        # Compiled only for the predict() call; training uses a custom loop
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    # ------------------------------------------------------------------
    # Override fit() to inject the distillation training loop
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
    ) -> TrainResult:
        import tensorflow as tf

        X_train = self._prepare_input(X_train).astype(np.float32)
        X_val   = self._prepare_input(X_val).astype(np.float32)
        n_classes = len(label_names)

        # ── Load teacher (frozen) ─────────────────────────────────────
        logger.info("Loading teacher model from %s", self.teacher_model_path)
        teacher = tf.keras.models.load_model(self.teacher_model_path)
        teacher.trainable = False

        # Pre-compute soft targets over training set
        logger.info("Computing teacher soft targets (T=%.1f) …", self.temperature)
        teacher_logits = self._get_logits(teacher, X_train)  # (N, C)
        soft_targets   = tf.nn.softmax(teacher_logits / self.temperature).numpy()

        # ── Build student ─────────────────────────────────────────────
        self._model = self._build_model(X_train.shape[1:], n_classes)
        norm_layer  = self._model.layers[1]
        if hasattr(norm_layer, "adapt"):
            norm_layer.adapt(X_train)

        optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        # ── Custom training loop ──────────────────────────────────────
        T      = tf.constant(self.temperature, dtype=tf.float32)
        alpha  = tf.constant(self.alpha,       dtype=tf.float32)
        n_val  = len(X_val)
        best_val_loss  = float("inf")
        no_improve     = 0
        patience       = 10
        patience_lr    = 5
        lr_factor      = 0.5
        min_lr         = 1e-6
        best_weights   = self._model.get_weights()

        dataset = tf.data.Dataset.from_tensor_slices(
            (X_train, y_train.astype(np.int32), soft_targets.astype(np.float32))
        ).shuffle(len(X_train)).batch(self.batch_size)

        val_ds = tf.data.Dataset.from_tensor_slices(
            (X_val, y_val.astype(np.int32))
        ).batch(self.batch_size)

        @tf.function
        def train_step(xb, yb, soft_b):
            with tf.GradientTape() as tape:
                logits = self._model(xb, training=True)
                # Hard-label cross-entropy
                hard_loss = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(
                        yb, tf.nn.softmax(logits), from_logits=False
                    )
                )
                # Soft KL divergence
                student_soft = tf.nn.log_softmax(logits / T)
                kl_loss = tf.reduce_mean(
                    tf.reduce_sum(soft_b * (tf.math.log(soft_b + 1e-8) - student_soft), axis=-1)
                ) * T * T
                loss = alpha * kl_loss + (1.0 - alpha) * hard_loss
            grads = tape.gradient(loss, self._model.trainable_variables)
            optimizer.apply_gradients(zip(grads, self._model.trainable_variables))
            return loss, hard_loss

        @tf.function
        def val_step(xb, yb):
            logits = self._model(xb, training=False)
            loss   = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    yb, tf.nn.softmax(logits), from_logits=False
                )
            )
            preds  = tf.argmax(logits, axis=1, output_type=tf.int32)
            return loss, preds

        no_improve_lr = 0
        for epoch in range(self.epochs):
            # Training
            epoch_loss = 0.0
            n_batches  = 0
            for xb, yb, soft_b in dataset:
                loss, _ = train_step(xb, yb, soft_b)
                epoch_loss += float(loss)
                n_batches  += 1
            epoch_loss /= max(n_batches, 1)

            # Validation
            val_loss   = 0.0
            n_correct  = 0
            n_val_batches = 0
            for xb, yb in val_ds:
                bloss, preds = val_step(xb, yb)
                val_loss    += float(bloss)
                n_correct   += int(tf.reduce_sum(tf.cast(preds == yb, tf.int32)).numpy())
                n_val_batches += 1
            val_loss /= max(n_val_batches, 1)
            val_acc   = n_correct / n_val

            logger.info(
                "[distillation_cnn] Epoch %3d/%d  loss=%.4f  val_loss=%.4f  val_acc=%.4f",
                epoch + 1, self.epochs, epoch_loss, val_loss, val_acc,
            )

            if mlflow_run is not None:
                import mlflow
                mlflow.log_metrics({
                    "loss": epoch_loss, "val_loss": val_loss, "val_accuracy": val_acc
                }, step=epoch)

            # Early stopping / LR decay
            if val_loss < best_val_loss - 1e-5:
                best_val_loss  = val_loss
                best_weights   = self._model.get_weights()
                no_improve     = 0
                no_improve_lr  = 0
            else:
                no_improve    += 1
                no_improve_lr += 1
                if no_improve_lr >= patience_lr:
                    current_lr = float(optimizer.learning_rate)
                    new_lr = max(current_lr * lr_factor, min_lr)
                    optimizer.learning_rate.assign(new_lr)
                    logger.info("[distillation_cnn] LR reduced to %.2e", new_lr)
                    no_improve_lr = 0
                if no_improve >= patience:
                    logger.info("[distillation_cnn] Early stopped at epoch %d", epoch + 1)
                    break

        self._model.set_weights(best_weights)

        # Add softmax for inference / evaluation
        inputs  = self._model.input
        outputs = tf.keras.layers.Softmax()(self._model.output)
        self._model = tf.keras.Model(inputs, outputs)

        # ── Evaluate and save (reuse KerasTrainer helpers) ────────────
        y_pred_val  = np.argmax(self._model.predict(X_val, verbose=0), axis=1)
        val_metrics = compute_metrics(y_val, y_pred_val, label_names=label_names)

        output_dir.mkdir(parents=True, exist_ok=True)
        model_path   = output_dir / "model.keras"
        self._model.save(model_path)
        model_size_kb = (
            sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file()) / 1024
            if model_path.is_dir() else model_path.stat().st_size / 1024
        )

        params = {
            "model":              self.name,
            "teacher_model_path": self.teacher_model_path,
            "epochs":             self.epochs,
            "batch_size":         self.batch_size,
            "dropout":            self.dropout,
            "learning_rate":      self.learning_rate,
        }
        params.update({k: str(v) for k, v in self._architecture_params().items()})

        save_classification_report(y_val, y_pred_val, label_names,
                                   output_dir / "classification_report.txt")
        save_confusion_matrix_png(val_metrics.get("confusion_matrix", []),
                                  label_names, output_dir / "confusion_matrix.png")
        save_model_info(output_dir, self.name, run_name, val_metrics, params, model_size_kb)

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

    @staticmethod
    def _get_logits(teacher, X: np.ndarray) -> np.ndarray:
        """Extract pre-softmax logits from the teacher.

        If the last layer is a softmax activation the penultimate layer's
        output is used instead; otherwise the raw output is returned.
        """
        import tensorflow as tf

        last = teacher.layers[-1]
        if (
            isinstance(last, tf.keras.layers.Activation) and last.get_config().get("activation") == "softmax"
            or isinstance(last, tf.keras.layers.Dense) and last.get_config().get("activation") == "softmax"
            or isinstance(last, tf.keras.layers.Softmax)
        ):
            logit_model = tf.keras.Model(teacher.input, teacher.layers[-2].output)
        else:
            logit_model = teacher
        return logit_model.predict(X, verbose=0)

    @classmethod
    def load(cls, path: Path) -> "DistillationCNNTrainer":
        import tensorflow as tf
        inst = cls.__new__(cls)
        inst._model = tf.keras.models.load_model(path)
        return inst

