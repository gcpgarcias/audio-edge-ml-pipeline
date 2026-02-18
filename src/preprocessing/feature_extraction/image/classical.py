"""
Classical image feature extractor.

Produces a fixed-length flat feature vector from a raster image (PNG, JPEG,
etc.), suitable for traditional ML classifiers and clustering algorithms:

    SVM · LDA · PCA · Decision Tree · Random Forest · K-NN · K-Means

Feature groups
--------------
Group                   Dimension (default)   Description
----------------------- --------------------- --------------------------------
HOG                     varies with resize_to  Histogram of Oriented Gradients
LBP histogram           26                     Local Binary Pattern histogram
                                               (P=24, R=3, uniform mapping)
Grayscale histogram     n_hist_bins (64)       Global intensity distribution
GLCM texture            6                      Gray-Level Co-occurrence Matrix
                                               properties averaged over 4 angles
                                               (contrast, dissimilarity,
                                               homogeneity, energy,
                                               correlation, ASM)
----------------------- --------------------- --------------------------------

With the default resize_to=(128, 128) the HOG contributes 8 100 values,
giving a total vector length of ≈ 8 196.

The extractor optionally crops the image to a normalised bounding box
(YOLOv8 format: x_center, y_center, width, height all in [0, 1]) before
computing features.  This is used by :class:`BIRDeepImageLoader` to isolate
individual bird-call detections from full-spectrogram images.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern

from ..base import BaseFeatureExtractor
from ..registry import register


def _load_gray(
    path: Path,
    resize_to: tuple[int, int],
    bbox_norm: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Load image as uint8 grayscale, optionally crop to a normalised bbox.

    Parameters
    ----------
    path:
        Image file path.
    resize_to:
        ``(width, height)`` to resize to after cropping.
    bbox_norm:
        YOLOv8 normalised bbox ``[x_center, y_center, w, h]``.
        When present the region is cropped before resizing.

    Returns
    -------
    np.ndarray
        uint8 grayscale array of shape ``(height, width)``.
    """
    img = Image.open(path).convert("L")  # L = 8-bit grayscale

    if bbox_norm is not None:
        x_c, y_c, bw, bh = bbox_norm
        W, H = img.size
        x1 = max(0, int((x_c - bw / 2) * W))
        y1 = max(0, int((y_c - bh / 2) * H))
        x2 = min(W, int((x_c + bw / 2) * W))
        y2 = min(H, int((y_c + bh / 2) * H))
        # Guard against degenerate boxes
        if x2 > x1 and y2 > y1:
            img = img.crop((x1, y1, x2, y2))

    img = img.resize(resize_to, Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


@register
class ImageClassicalExtractor(BaseFeatureExtractor):
    """Classical image features (HOG + LBP + histogram + GLCM texture).

    Parameters
    ----------
    resize_to:
        ``(width, height)`` used for all internal computations.
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
        Number of bins in the grayscale intensity histogram.
    glcm_distances:
        Pixel-pair distances for GLCM computation.
    """

    name         = "image_classical"
    feature_type = "classical"
    modality     = "image"

    def __init__(
        self,
        resize_to:           tuple[int, int] = (128, 128),
        hog_orientations:    int             = 9,
        hog_pixels_per_cell: tuple[int, int] = (8, 8),
        hog_cells_per_block: tuple[int, int] = (2, 2),
        lbp_n_points:        int             = 24,
        lbp_radius:          float           = 3.0,
        n_hist_bins:         int             = 64,
        glcm_distances:      list[int]       = (1,),
    ) -> None:
        self.resize_to            = resize_to
        self.hog_orientations     = hog_orientations
        self.hog_pixels_per_cell  = hog_pixels_per_cell
        self.hog_cells_per_block  = hog_cells_per_block
        self.lbp_n_points         = lbp_n_points
        self.lbp_radius           = lbp_radius
        self.n_hist_bins          = n_hist_bins
        self.glcm_distances       = list(glcm_distances)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract(
        self,
        sample_path: Path,
        bbox_norm:   Optional[Sequence[float]] = None,
        **_kwargs,
    ) -> np.ndarray:
        """Extract classical features from *sample_path*.

        Parameters
        ----------
        sample_path:
            Path to a raster image (PNG, JPEG, …).
        bbox_norm:
            YOLOv8 normalised bounding box ``[x_center, y_center, w, h]``
            used to crop the image before extraction.  *None* → full image.

        Returns
        -------
        np.ndarray
            1-D float32 feature vector.
        """
        gray = _load_gray(sample_path, self.resize_to, bbox_norm)
        return self._compute_features(gray).astype(np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_features(self, gray: np.ndarray) -> np.ndarray:
        parts: list[np.ndarray] = []

        # ---- HOG -------------------------------------------------------
        hog_feats = hog(
            gray,
            orientations=self.hog_orientations,
            pixels_per_cell=self.hog_pixels_per_cell,
            cells_per_block=self.hog_cells_per_block,
            feature_vector=True,
        )
        parts.append(hog_feats)

        # ---- LBP histogram ---------------------------------------------
        lbp = local_binary_pattern(
            gray,
            P=self.lbp_n_points,
            R=self.lbp_radius,
            method="uniform",
        )
        n_patterns = self.lbp_n_points + 2  # uniform LBP has P+2 unique codes
        lbp_hist, _ = np.histogram(
            lbp,
            bins=n_patterns,
            range=(0, n_patterns),
            density=True,
        )
        parts.append(lbp_hist.astype(np.float32))

        # ---- Grayscale intensity histogram -----------------------------
        g_hist, _ = np.histogram(gray, bins=self.n_hist_bins, range=(0, 256),
                                  density=True)
        parts.append(g_hist.astype(np.float32))

        # ---- GLCM texture features ------------------------------------
        # Four angles: 0°, 45°, 90°, 135°
        angles  = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        glcm    = graycomatrix(
            gray,
            distances=self.glcm_distances,
            angles=angles,
            levels=256,
            symmetric=True,
            normed=True,
        )
        props   = ("contrast", "dissimilarity", "homogeneity",
                   "energy", "correlation", "ASM")
        glcm_feats = np.array(
            [graycoprops(glcm, prop).mean() for prop in props],
            dtype=np.float32,
        )
        parts.append(glcm_feats)

        return np.concatenate(parts)