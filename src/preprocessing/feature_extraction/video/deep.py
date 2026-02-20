"""
Deep video feature extractors.

Each extractor preserves the temporal structure required by sequence and
volumetric deep learning models.

Extractor                  name                      Output shape      Suited for
-------------------------  ------------------------  ----------------  --------------------------------
VideoFrameSequence         video_frame_seq           (T, H, W, C)      3D CNN, CNN+RNN, ViT
VideoMobileNetV2Sequence   video_mobilenet_v2_seq    (T, 1280)         LSTM, Transformer on embeddings

Both extractors sample *max_frames* evenly spaced frames from the video using
``cv2.VideoCapture`` (OpenCV), applying the same ``_open_and_sample`` helper
defined in the sibling ``classical`` module.

``VideoMobileNetV2Sequence`` requires TensorFlow (already a project dependency).
The MobileNetV2 backbone is loaded lazily on first call and cached as an
instance attribute, matching the pattern used in ``image/deep.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..base import BaseFeatureExtractor
from ..registry import register
from .classical import _open_and_sample  # shared frame-sampling helper


# ---------------------------------------------------------------------------
# Raw frame sequence
# ---------------------------------------------------------------------------

@register
class VideoFrameSequence(BaseFeatureExtractor):
    """Normalised frame sequence tensor.

    Shape: ``(T, H, W, C)`` — T sampled frames, each an ``(H, W, C)`` float32
    array with pixel values normalised to ``[0, 1]``.

    - ``as_gray=False`` (default) → C = 3 (RGB).
    - ``as_gray=True`` → C = 1 (grayscale, channel dim retained for
      framework compatibility).

    The number of output frames ``T`` equals ``min(max_frames, actual_frames)``.

    This is the natural input to:
    - **3D CNN** — treats the sequence as a volumetric tensor.
    - **CNN + RNN** — CNN encodes each frame; RNN processes the sequence.
    - **Video Transformer** — patches across space and time.

    Parameters
    ----------
    resize_to:
        ``(width, height)`` of each output frame.
    max_frames:
        Maximum number of evenly spaced frames to sample per video.
    as_gray:
        If *True*, output shape is ``(T, H, W, 1)``; otherwise ``(T, H, W, 3)``.
    """

    name         = "video_frame_seq"
    feature_type = "deep"
    modality     = "video"

    def __init__(
        self,
        resize_to:  tuple[int, int] = (128, 128),
        max_frames: int             = 16,
        as_gray:    bool            = False,
    ) -> None:
        self.resize_to  = resize_to
        self.max_frames = max_frames
        self.as_gray    = as_gray

    def extract(
        self,
        sample_path: Path,
        **_kwargs,
    ) -> np.ndarray:
        """Return a normalised frame sequence for *sample_path*.

        Parameters
        ----------
        sample_path:
            Path to a video file (.mp4, .avi, .mov, etc.).

        Returns
        -------
        np.ndarray
            float32 array of shape ``(T, H, W, C)``, values in ``[0, 1]``.
        """
        frames = _open_and_sample(
            sample_path, self.max_frames, self.resize_to, as_gray=self.as_gray
        )
        # Stack → (T, H, W) or (T, H, W, 3)
        arr = np.stack(frames, axis=0).astype(np.float32) / 255.0

        if self.as_gray:
            arr = arr[:, :, :, np.newaxis]  # add channel dim → (T, H, W, 1)

        return arr  # (T, H, W, C)


# ---------------------------------------------------------------------------
# MobileNetV2 per-frame embedding sequence
# ---------------------------------------------------------------------------

@register
class VideoMobileNetV2Sequence(BaseFeatureExtractor):
    """Per-frame MobileNetV2 embedding sequence.

    Shape: ``(T, 1280)``

    Each sampled frame is passed through a MobileNetV2 backbone (pretrained
    on ImageNet, global-average-pooled) to produce a 1 280-dimensional
    embedding.  The result is a 2-D tensor of shape ``(T, 1280)`` suitable for
    sequence models:

    - **LSTM / GRU** — treat each frame embedding as a time step.
    - **Transformer** — attend across frame embeddings.
    - **Temporal pooling + classifier** — mean/max-pool over T, then MLP.

    The MobileNetV2 backbone is instantiated once per extractor instance and
    reused across calls to avoid redundant loading.

    Parameters
    ----------
    input_size:
        ``(width, height)`` fed to MobileNetV2.  Must be ≥ 32.
        Defaults to ``(224, 224)`` (original ImageNet training size).
    max_frames:
        Maximum number of evenly spaced frames to sample per video.
    trainable:
        If *False* (default), backbone weights are frozen.
    """

    name         = "video_mobilenet_v2_seq"
    feature_type = "deep"
    modality     = "video"

    def __init__(
        self,
        input_size: tuple[int, int] = (224, 224),
        max_frames: int             = 16,
        trainable:  bool            = False,
    ) -> None:
        self.input_size = input_size
        self.max_frames = max_frames
        self.trainable  = trainable
        self._model     = None  # lazy init

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract(
        self,
        sample_path: Path,
        **_kwargs,
    ) -> np.ndarray:
        """Return a ``(T, 1280)`` MobileNetV2 embedding sequence.

        Parameters
        ----------
        sample_path:
            Path to a video file.

        Returns
        -------
        np.ndarray
            float32 array of shape ``(T, 1280)``.
        """
        import tensorflow as tf

        model = self._get_model()

        frames = _open_and_sample(
            sample_path, self.max_frames, self.input_size, as_gray=False
        )

        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

        embeddings: list[np.ndarray] = []
        for frame in frames:
            arr = frame.astype(np.float32)
            arr = preprocess(arr)                       # scale to [-1, 1]
            batch = arr[np.newaxis, ...]               # (1, H, W, 3)
            emb = model(batch, training=False).numpy().squeeze(0)  # (1280,)
            embeddings.append(emb.astype(np.float32))

        return np.stack(embeddings, axis=0)  # (T, 1280)

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
            pooling="avg",      # global average pool → (1280,)
            weights="imagenet",
        )
        backbone.trainable = self.trainable
        self._model = backbone
        return self._model