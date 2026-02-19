"""
Core abstractions for the feature extraction layer.

Provides:
- FeatureSet: uniform container compatible with supervised, unsupervised, and
  semi-supervised workflows.
- BaseFeatureExtractor: abstract base class for all feature extractors.
- BaseDatasetLoader: abstract base class for all dataset loaders.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional

import numpy as np

if TYPE_CHECKING:
    import tensorflow as tf

logger = logging.getLogger(__name__)


@dataclass
class FeatureSet:
    """Uniform container for extracted features.

    Supports:
    - Supervised learning:    labels and label_names are populated.
    - Unsupervised learning:  labels/label_names are None; cluster_assignments
                              can be populated after fitting (e.g. K-Means).
    - Semi-supervised:        labels uses -1 for unlabelled samples (sklearn
                              convention for LabelSpreading / LabelPropagation).
    """

    # --- Required fields ---
    features:     np.ndarray   # shape (N, *feature_dims)
    feature_type: str          # "classical" | "deep"
    modality:     str          # "audio" | "image" | "text" | "video"
    metadata:     list[dict]   # per-sample metadata dicts

    # --- Optional fields (None → unsupervised) ---
    labels:              Optional[np.ndarray] = None  # integer class indices
    label_names:         Optional[list[str]]  = None  # index → class name
    cluster_assignments: Optional[np.ndarray] = None  # set after unsupervised fit

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_samples(self) -> int:
        return len(self.features)

    @property
    def feature_shape(self) -> tuple:
        return self.features.shape[1:]

    @property
    def is_supervised(self) -> bool:
        return self.labels is not None

    @property
    def n_classes(self) -> Optional[int]:
        if self.label_names is not None:
            return len(self.label_names)
        if self.labels is not None:
            return int(self.labels.max()) + 1
        return None

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def to_sklearn(self) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Return (X, y) suitable for sklearn estimators.

        y is:
        - ground-truth class indices when labels are present,
        - cluster assignment indices when only cluster_assignments exist,
        - None for purely unsupervised use (caller passes X only).
        """
        if self.labels is not None:
            return self.features, self.labels
        if self.cluster_assignments is not None:
            return self.features, self.cluster_assignments
        return self.features, None

    def to_tf_dataset(
        self,
        batch_size: int = 32,
        shuffle: bool = False,
        seed: Optional[int] = None,
    ) -> "tf.data.Dataset":
        """Return a batched, prefetched tf.data.Dataset.

        Yields (features, labels) tuples when labels are present, or plain
        feature tensors otherwise.
        """
        import tensorflow as tf  # lazy import – optional dependency

        if self.labels is not None:
            ds = tf.data.Dataset.from_tensor_slices(
                (self.features.astype(np.float32), self.labels.astype(np.int32))
            )
        else:
            ds = tf.data.Dataset.from_tensor_slices(
                self.features.astype(np.float32)
            )
        if shuffle:
            ds = ds.shuffle(buffer_size=self.n_samples, seed=seed)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        label_info = (
            f"labels={self.n_classes} classes"
            if self.is_supervised
            else "unsupervised"
        )
        return (
            f"FeatureSet("
            f"modality={self.modality!r}, "
            f"feature_type={self.feature_type!r}, "
            f"n_samples={self.n_samples}, "
            f"feature_shape={self.feature_shape}, "
            f"{label_info})"
        )


class BaseFeatureExtractor(ABC):
    """Abstract base class for all feature extractors.

    Subclasses must declare three class-level attributes:
        name         str  – unique key used by the registry
        feature_type str  – "classical" or "deep"
        modality     str  – "audio", "image", "text", or "video"

    and implement:
        extract(sample_path, **kwargs) -> np.ndarray
    """

    name:         str  # set by concrete subclass
    feature_type: str
    modality:     str

    @abstractmethod
    def extract(self, sample_path: Optional[Path], **kwargs) -> np.ndarray:
        """Extract features from a single sample.

        Parameters
        ----------
        sample_path:
            Path to the raw sample file, or *None* for in-memory samples
            (e.g. individual rows from a tabular dataset, or text documents
            stored inside *kwargs* by a JSON/CSV loader).
        **kwargs:
            Per-sample metadata forwarded from the dataset loader (e.g.
            ``start_time`` / ``end_time`` for audio, ``bbox_norm`` for image,
            ``text`` for in-memory text documents, or column values for
            tabular rows).

        Returns
        -------
        np.ndarray
            Feature array for this sample.  Shape is extractor-specific.
        """
        ...

    def extract_dataset(
        self,
        loader: "BaseDatasetLoader",
        max_samples: Optional[int] = None,
    ) -> FeatureSet:
        """Extract features for every sample yielded by *loader*.

        Parameters
        ----------
        loader:
            Any :class:`BaseDatasetLoader` instance.
        max_samples:
            Stop after this many samples (useful for quick smoke tests).

        Returns
        -------
        FeatureSet
        """
        all_features: list[np.ndarray] = []
        all_labels:   list[int]        = []
        all_meta:     list[dict]       = []
        label_to_idx: dict[str, int]   = {}

        for i, (sample_path, label, meta) in enumerate(loader):
            if max_samples is not None and i >= max_samples:
                break
            try:
                feat = self.extract(sample_path, **meta)
            except Exception as exc:
                logger.warning("Skipping %s: %s", sample_path, exc)
                continue

            all_features.append(feat)
            all_meta.append(meta)

            if label is not None:
                if label not in label_to_idx:
                    label_to_idx[label] = len(label_to_idx)
                all_labels.append(label_to_idx[label])

        if not all_features:
            raise RuntimeError("No features were successfully extracted.")

        features = np.stack(all_features)
        labels = np.array(all_labels, dtype=np.int32) if all_labels else None
        label_names = (
            [k for k, _ in sorted(label_to_idx.items(), key=lambda x: x[1])]
            if label_to_idx
            else None
        )

        return FeatureSet(
            features=features,
            feature_type=self.feature_type,
            modality=self.modality,
            metadata=all_meta,
            labels=labels,
            label_names=label_names,
        )


class BaseDatasetLoader(ABC):
    """Abstract base class for dataset loaders.

    Iterating over a loader yields ``(sample_path, label, metadata)`` tuples:

    * ``sample_path`` – :class:`~pathlib.Path` to the raw file, or *None* for
                        in-memory samples (tabular rows, JSON text documents).
    * ``label``       – string class label, or *None* for unlabelled samples.
    * ``metadata``    – arbitrary dict forwarded to the feature extractor as
                        keyword arguments (e.g. ``start_time`` / ``end_time``
                        for audio, ``bbox_norm`` for image, ``text`` for
                        in-memory text, or column key-value pairs for tabular).
    """

    @abstractmethod
    def __iter__(self) -> Iterator[tuple[Optional[Path], Optional[str], dict]]:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...