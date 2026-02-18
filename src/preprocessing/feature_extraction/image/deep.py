"""
Deep image feature extractors.

Each extractor preserves spatial or embedding structure required by deep
learning models rather than collapsing into a flat vector.

Extractor           name                  Output shape        Suited for
------------------  --------------------  ------------------  ---------------------
ImagePixels         image_pixels          (H, W, C)           CNN, ViT
ImageMobileNetV2    image_mobilenet_v2    (1280,)             Transfer learning,
                                                              KNN, SVM on embeddings

Both extractors accept an optional ``bbox_norm`` metadata key
``[x_center, y_center, w, h]`` (normalised [0,1]) so that they can be
combined with :class:`BIRDeepImageLoader` for patch-level feature extraction.

``ImageMobileNetV2`` requires TensorFlow (already a project dependency).
The backbone is loaded lazily and cached as a class-level singleton so that
iterating over a dataset does not reload it on every call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image

from ..base import BaseFeatureExtractor
from ..registry import register


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_rgb(
    path:      Path,
    resize_to: tuple[int, int],
    bbox_norm: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Load image as uint8 RGB, optionally crop to a normalised bbox.

    Parameters
    ----------
    path:
        Image file path.
    resize_to:
        ``(width, height)`` applied after any crop.
    bbox_norm:
        YOLOv8 normalised bbox ``[x_center, y_center, w, h]``.

    Returns
    -------
    np.ndarray
        uint8 RGB array of shape ``(height, width, 3)``.
    """
    img = Image.open(path).convert("RGB")

    if bbox_norm is not None:
        x_c, y_c, bw, bh = bbox_norm
        W, H = img.size
        x1 = max(0, int((x_c - bw / 2) * W))
        y1 = max(0, int((y_c - bh / 2) * H))
        x2 = min(W, int((x_c + bw / 2) * W))
        y2 = min(H, int((y_c + bh / 2) * H))
        if x2 > x1 and y2 > y1:
            img = img.crop((x1, y1, x2, y2))

    img = img.resize(resize_to, Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Raw pixels
# ---------------------------------------------------------------------------

@register
class ImagePixels(BaseFeatureExtractor):
    """Normalised raw pixel array.

    Shape: ``(height, width, channels)``

    Pixels are normalised to ``[0, 1]`` (float32).  The number of channels
    follows the *as_gray* flag: 1 for grayscale, 3 for RGB.

    This is the simplest deep representation and the natural input to a
    custom CNN or Vision Transformer trained from scratch.

    Parameters
    ----------
    resize_to:
        ``(width, height)`` of the output array.
    as_gray:
        If *True*, convert to grayscale → output shape ``(H, W, 1)``.
        If *False* (default), keep RGB → output shape ``(H, W, 3)``.
    """

    name         = "image_pixels"
    feature_type = "deep"
    modality     = "image"

    def __init__(
        self,
        resize_to: tuple[int, int] = (128, 128),
        as_gray:   bool            = False,
    ) -> None:
        self.resize_to = resize_to
        self.as_gray   = as_gray

    def extract(
        self,
        sample_path: Path,
        bbox_norm:   Optional[Sequence[float]] = None,
        **_kwargs,
    ) -> np.ndarray:
        """Return a normalised pixel array for *sample_path*.

        Parameters
        ----------
        sample_path:
            Path to a raster image.
        bbox_norm:
            Optional normalised crop box ``[x_center, y_center, w, h]``.
        """
        if self.as_gray:
            from PIL import Image as _Image
            img = _Image.open(sample_path).convert("L")
            if bbox_norm is not None:
                x_c, y_c, bw, bh = bbox_norm
                W, H = img.size
                x1 = max(0, int((x_c - bw / 2) * W))
                y1 = max(0, int((y_c - bh / 2) * H))
                x2 = min(W, int((x_c + bw / 2) * W))
                y2 = min(H, int((y_c + bh / 2) * H))
                if x2 > x1 and y2 > y1:
                    img = img.crop((x1, y1, x2, y2))
            img = img.resize(self.resize_to, Image.LANCZOS)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            return arr[:, :, np.newaxis]  # add channel dim → (H, W, 1)

        arr = _load_rgb(sample_path, self.resize_to, bbox_norm).astype(np.float32)
        return arr / 255.0  # (H, W, 3)


# ---------------------------------------------------------------------------
# MobileNetV2 embeddings
# ---------------------------------------------------------------------------

@register
class ImageMobileNetV2(BaseFeatureExtractor):
    """MobileNetV2 global-average-pooled embedding vector.

    Shape: ``(1280,)``

    Uses the MobileNetV2 backbone pretrained on ImageNet, with the
    classification head removed and global average pooling applied.  The
    result is a 1 280-dimensional embedding suitable for downstream
    classifiers (SVM, K-NN) or as input to a lightweight head network.

    The backbone is instantiated once per extractor instance and reused
    across calls.  TensorFlow is a mandatory project dependency and is
    imported lazily (first call) to keep module import time low.

    Parameters
    ----------
    input_size:
        ``(width, height)`` fed to MobileNetV2.  Must be ≥ 32.
        The default ``(224, 224)`` matches the original ImageNet training.
    trainable:
        If *False* (default), backbone weights are frozen.  Setting this to
        *True* is useful if you intend to fine-tune the extractor as part of
        a larger TF model.
    """

    name         = "image_mobilenet_v2"
    feature_type = "deep"
    modality     = "image"

    def __init__(
        self,
        input_size: tuple[int, int] = (224, 224),
        trainable:  bool            = False,
    ) -> None:
        self.input_size = input_size
        self.trainable  = trainable
        self._model     = None  # lazy init

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract(
        self,
        sample_path: Path,
        bbox_norm:   Optional[Sequence[float]] = None,
        **_kwargs,
    ) -> np.ndarray:
        """Return a 1 280-dim MobileNetV2 embedding for *sample_path*.

        Parameters
        ----------
        sample_path:
            Path to a raster image.
        bbox_norm:
            Optional normalised crop box ``[x_center, y_center, w, h]``.
        """
        model = self._get_model()

        # Load, optionally crop, and resize to model input size
        arr = _load_rgb(sample_path, self.input_size, bbox_norm).astype(np.float32)

        # MobileNetV2 expects pixel values in [-1, 1]
        import tensorflow as tf
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)

        # Add batch dimension → (1, H, W, 3)
        batch = arr[np.newaxis, ...]
        embedding = model(batch, training=False).numpy().squeeze(0)  # (1280,)
        return embedding.astype(np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_model(self):
        """Lazily build (or return cached) the MobileNetV2 feature extractor."""
        if self._model is not None:
            return self._model

        import tensorflow as tf

        W, H = self.input_size
        backbone = tf.keras.applications.MobileNetV2(
            input_shape=(H, W, 3),
            include_top=False,
            pooling="avg",          # global average pool → (1280,)
            weights="imagenet",
        )
        backbone.trainable = self.trainable
        self._model = backbone
        return self._model