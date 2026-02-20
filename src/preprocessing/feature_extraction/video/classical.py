"""
Classical video feature extractor.

Produces a fixed-length flat feature vector from a video file, suitable for
traditional ML classifiers and clustering algorithms:

    SVM · LDA · PCA · Decision Tree · Random Forest · K-NN · K-Means

Strategy
--------
1. Open the video with ``cv2.VideoCapture`` and sample *max_frames* evenly
   spaced frames.
2. For each frame (resized to *resize_to*, converted to grayscale):
   - HOG (Histogram of Oriented Gradients) with coarse cell size → compact dim.
   - LBP histogram (uniform, 26 bins).
   - Grayscale intensity histogram (64 bins).
3. Aggregate per-frame vectors with **mean + std** over the time axis →
   flat float32 vector of length ``2 × per_frame_dim``.
4. If *optical_flow* is *True*, compute dense Farneback optical flow between
   consecutive sampled frames and append global flow statistics:
   - magnitude mean, magnitude std (2 dims).
   - 8-bin direction histogram (8 dims).
   → 10 extra dimensions per transition, averaged over all frame pairs.

Feature dimensions (default settings, resize_to=(128,128))
-----------------------------------------------------------
HOG (9 orient, 16×16 cell, 2×2 block)  →  441  dims  (7×7×4×9 ÷ 2-overlap)
LBP histogram                           →   26  dims
Grayscale histogram                     →   64  dims
Per-frame total                         →  531  dims
Mean + std (×2)                         → 1062  dims
+ optical_flow=True                     →  +10  dims
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from skimage.feature import hog, local_binary_pattern

from ..base import BaseFeatureExtractor
from ..registry import register

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared frame-sampling helper
# ---------------------------------------------------------------------------

def _open_and_sample(
    video_path: Path,
    max_frames: int,
    resize_to:  tuple[int, int],
    as_gray:    bool = True,
) -> list[np.ndarray]:
    """Open a video and return up to *max_frames* evenly spaced frames.

    Parameters
    ----------
    video_path:
        Path to the video file.
    max_frames:
        Maximum number of frames to return.
    resize_to:
        ``(width, height)`` to resize each frame to.
    as_gray:
        If *True*, convert BGR frames to grayscale (uint8).
        If *False*, convert BGR to RGB (uint8).

    Returns
    -------
    list[np.ndarray]
        Frames as uint8 arrays, shape ``(H, W)`` or ``(H, W, 3)``.

    Raises
    ------
    RuntimeError
        If the video cannot be opened or has no readable frames.
    """
    import cv2  # lazy import — opencv-python must be installed

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        # Some containers don't report frame count; fall back to sequential read
        total = None

    W, H = resize_to

    if total is not None and total > 0:
        # Evenly spaced indices
        indices = set(
            int(round(i * (total - 1) / max(max_frames - 1, 1)))
            for i in range(min(max_frames, total))
        )
        frames: list[np.ndarray] = []
        for idx in sorted(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.resize(frame, (W, H))
            if as_gray:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    else:
        # Sequential fallback
        frames = []
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (W, H))
            if as_gray:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames could be read from: {video_path}")

    return frames


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

@register
class VideoClassicalExtractor(BaseFeatureExtractor):
    """Classical video features (HOG + LBP + histogram + optional optical flow).

    The final feature vector has length ``2 × per_frame_dim [+ 10]``:

    - ``per_frame_dim = hog_dim + 26 + n_hist_bins``
    - Mean and standard deviation are computed over all sampled frames.
    - When *optical_flow=True*, 10 flow statistics are appended (see module
      docstring).

    Parameters
    ----------
    resize_to:
        ``(width, height)`` used for internal computation.
    max_frames:
        Maximum number of evenly spaced frames to sample per video.
    hog_orientations:
        Number of gradient orientations for HOG.
    hog_pixels_per_cell:
        Cell size ``(rows, cols)`` for HOG.
    hog_cells_per_block:
        Block size in cells ``(rows, cols)`` for HOG normalisation.
    lbp_n_points:
        Number of circularly symmetric LBP neighbour points.
    lbp_radius:
        LBP neighbourhood radius in pixels.
    n_hist_bins:
        Number of bins in the per-frame grayscale intensity histogram.
    optical_flow:
        If *True*, compute dense Farneback optical flow between consecutive
        sampled frames and append 10 global flow statistics.
    """

    name         = "video_classical"
    feature_type = "classical"
    modality     = "video"

    def __init__(
        self,
        resize_to:           tuple[int, int] = (128, 128),
        max_frames:          int             = 16,
        hog_orientations:    int             = 9,
        hog_pixels_per_cell: tuple[int, int] = (16, 16),
        hog_cells_per_block: tuple[int, int] = (2, 2),
        lbp_n_points:        int             = 24,
        lbp_radius:          float           = 3.0,
        n_hist_bins:         int             = 64,
        optical_flow:        bool            = False,
    ) -> None:
        self.resize_to            = resize_to
        self.max_frames           = max_frames
        self.hog_orientations     = hog_orientations
        self.hog_pixels_per_cell  = hog_pixels_per_cell
        self.hog_cells_per_block  = hog_cells_per_block
        self.lbp_n_points         = lbp_n_points
        self.lbp_radius           = lbp_radius
        self.n_hist_bins          = n_hist_bins
        self.optical_flow         = optical_flow

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract(
        self,
        sample_path: Path,
        **_kwargs,
    ) -> np.ndarray:
        """Extract classical features from the video at *sample_path*.

        Parameters
        ----------
        sample_path:
            Path to a video file (.mp4, .avi, etc.).

        Returns
        -------
        np.ndarray
            1-D float32 feature vector of length ``2 × per_frame_dim [+ 10]``.
        """
        frames = _open_and_sample(sample_path, self.max_frames, self.resize_to, as_gray=True)
        per_frame = np.stack([self._frame_features(f) for f in frames], axis=0)  # (T, D)

        parts: list[np.ndarray] = [
            per_frame.mean(axis=0).astype(np.float32),
            per_frame.std(axis=0).astype(np.float32),
        ]

        if self.optical_flow and len(frames) >= 2:
            parts.append(self._flow_features(frames))

        return np.concatenate(parts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _frame_features(self, gray: np.ndarray) -> np.ndarray:
        """Compute HOG + LBP + histogram for a single grayscale frame."""
        parts: list[np.ndarray] = []

        # HOG
        hog_feats = hog(
            gray,
            orientations=self.hog_orientations,
            pixels_per_cell=self.hog_pixels_per_cell,
            cells_per_block=self.hog_cells_per_block,
            feature_vector=True,
        )
        parts.append(hog_feats.astype(np.float32))

        # LBP histogram (uniform mapping, P+2 unique codes)
        lbp = local_binary_pattern(
            gray, P=self.lbp_n_points, R=self.lbp_radius, method="uniform"
        )
        n_patterns = self.lbp_n_points + 2
        lbp_hist, _ = np.histogram(
            lbp, bins=n_patterns, range=(0, n_patterns), density=True
        )
        parts.append(lbp_hist.astype(np.float32))

        # Grayscale intensity histogram
        g_hist, _ = np.histogram(gray, bins=self.n_hist_bins, range=(0, 256), density=True)
        parts.append(g_hist.astype(np.float32))

        return np.concatenate(parts)

    def _flow_features(self, frames: list[np.ndarray]) -> np.ndarray:
        """Compute mean optical-flow statistics across consecutive frame pairs.

        Returns a 10-dim vector:
        ``[mag_mean, mag_std, angle_hist×8]``.
        """
        import cv2

        all_mag:   list[float] = []
        all_angle: list[np.ndarray] = []

        for prev, curr in zip(frames[:-1], frames[1:]):
            flow = cv2.calcOpticalFlowFarneback(
                prev, curr,
                None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2,
                flags=0,
            )
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            all_mag.append(float(mag.mean()))
            all_mag.append(float(mag.std()))
            hist, _ = np.histogram(ang, bins=8, range=(0, 2 * np.pi), density=True)
            all_angle.append(hist.astype(np.float32))

        mag_arr   = np.array(all_mag, dtype=np.float32).reshape(-1, 2)  # (N_pairs, 2)
        angle_arr = np.stack(all_angle, axis=0)                         # (N_pairs, 8)

        return np.concatenate([
            mag_arr.mean(axis=0),    # [mag_mean, mag_std] averaged over pairs
            angle_arr.mean(axis=0),  # 8-bin flow direction histogram
        ])