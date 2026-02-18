"""
BIRDeep Audio Annotations dataset loaders.

Two loaders are provided for the BIRDeep_AudioAnnotations dataset:

``BIRDeepLoader``
    Yields ``(wav_path, species_label, metadata)`` tuples.  Metadata carries
    ``start_time`` / ``end_time`` so that feature extractors can slice the
    relevant segment from the 1-minute recording.

``BIRDeepImageLoader``
    Yields ``(png_path, species_label, metadata)`` tuples where each PNG is
    the full 1-minute spectrogram image.  Metadata carries a ``bbox_norm``
    key — a normalised YOLOv8 bounding box ``[x_center, y_center, w, h]``
    that image feature extractors use to crop the bird-call region before
    computing features.

Both loaders read the same pre-split CSV files
(``train_file.csv``, ``test_file.csv``, ``validation_file.csv``,
``dataset.csv``) and produce one sample per annotation row.

Dataset layout (expected)
-------------------------
<dataset_root>/
    dataset.csv
    train_file.csv
    test_file.csv
    validation_file.csv
    Audios/
        <SITE>/<YYYY_MM_DD>/<SITE>_<YYYYMMDD>_<HHMMSS>.WAV
        Data Augmentation/<SITE>/<YYYY_MM_DD>/...
    images/
        <SITE>/<YYYY_MM_DD>/<SITE>_<YYYYMMDD>_<HHMMSS>.PNG
    labels/
        <SITE>/<YYYY_MM_DD>/<SITE>_<YYYYMMDD>_<HHMMSS>.txt
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from ..feature_extraction.base import BaseDatasetLoader

logger = logging.getLogger(__name__)

# Maps the 'split' argument to its CSV filename.
_SPLIT_FILES: dict[str, str] = {
    "train":      "train_file.csv",
    "test":       "test_file.csv",
    "validation": "validation_file.csv",
    "all":        "dataset.csv",
}


class BIRDeepLoader(BaseDatasetLoader):
    """Iterate over annotated bird-call segments in the BIRDeep dataset.

    Each annotation (one row in the CSV) corresponds to a single species
    vocalisation within a one-minute recording.  The loader yields the
    full WAV file path together with ``start_time`` / ``end_time`` metadata
    so that feature extractors can extract only the relevant segment.

    Parameters
    ----------
    dataset_root:
        Path to the BIRDeep_AudioAnnotations directory.
    split:
        One of ``"train"``, ``"test"``, ``"validation"``, or ``"all"``.
    audio_subdir:
        Sub-directory that contains the WAV files (default ``"Audios"``).
    include_augmented:
        If *False* (default), rows whose path begins with
        ``"Data Augmentation"`` are skipped.
    min_segment_duration:
        Annotations whose ``end_time - start_time`` is shorter than this
        value (seconds) are skipped.  Set to 0.0 to keep all.
    species_filter:
        If given, only annotations whose ``specie`` is in this collection
        are yielded.
    """

    def __init__(
        self,
        dataset_root:           Path | str,
        split:                  str  = "train",
        audio_subdir:           str  = "Audios",
        include_augmented:      bool = False,
        min_segment_duration:   float = 0.05,
        species_filter:         Optional[set[str]] = None,
    ) -> None:
        if split not in _SPLIT_FILES:
            raise ValueError(
                f"split must be one of {list(_SPLIT_FILES)}, got {split!r}."
            )

        self.dataset_root  = Path(dataset_root)
        self.audio_dir     = self.dataset_root / audio_subdir
        self.split         = split
        self.include_augmented    = include_augmented
        self.min_segment_duration = min_segment_duration
        self.species_filter       = species_filter

        csv_path = self.dataset_root / _SPLIT_FILES[split]
        if not csv_path.exists():
            raise FileNotFoundError(
                f"CSV file not found: {csv_path}. "
                "Ensure dataset_root points to BIRDeep_AudioAnnotations/."
            )

        self._df = self._load_and_filter(csv_path)
        logger.info(
            "BIRDeepLoader [%s] – %d annotations across %d unique recordings.",
            split,
            len(self._df),
            self._df["path"].nunique(),
        )

    # ------------------------------------------------------------------
    # BaseDatasetLoader interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._df)

    def __iter__(self) -> Iterator[tuple[Path, Optional[str], dict]]:
        for _, row in self._df.iterrows():
            audio_path = self.audio_dir / row["path"]
            if not audio_path.exists():
                logger.warning("Audio file not found, skipping: %s", audio_path)
                continue

            label = str(row["specie"])
            meta  = self._row_to_meta(row)
            yield audio_path, label, meta

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def species(self) -> list[str]:
        """Sorted list of unique species present in this split."""
        return sorted(self._df["specie"].unique().tolist())

    @property
    def n_species(self) -> int:
        return len(self.species)

    @property
    def recorders(self) -> list[str]:
        """Sorted list of recorder site IDs present in this split."""
        return sorted(self._df["recorder"].unique().tolist())

    def species_counts(self) -> "pd.Series":
        """Return annotation counts per species, descending."""
        return self._df["specie"].value_counts()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_and_filter(self, csv_path: Path) -> "pd.DataFrame":
        df = pd.read_csv(csv_path)

        # Normalise column names (strip whitespace)
        df.columns = df.columns.str.strip()

        # Coerce numeric columns
        for col in ("start_time", "end_time", "low_frequency", "high_frequency"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop rows with missing mandatory fields
        mandatory = ["path", "specie", "start_time", "end_time"]
        before = len(df)
        df = df.dropna(subset=mandatory)
        dropped = before - len(df)
        if dropped:
            logger.warning("Dropped %d rows with missing mandatory fields.", dropped)

        # Filter out augmented samples unless requested
        if not self.include_augmented:
            mask_aug = df["path"].str.startswith("Data Augmentation")
            n_aug = mask_aug.sum()
            if n_aug:
                logger.info("Skipping %d augmented annotations.", n_aug)
            df = df[~mask_aug]

        # Filter by minimum segment duration
        if self.min_segment_duration > 0.0:
            duration = df["end_time"] - df["start_time"]
            df = df[duration >= self.min_segment_duration]

        # Optional species filter
        if self.species_filter is not None:
            df = df[df["specie"].isin(self.species_filter)]

        return df.reset_index(drop=True)

    @staticmethod
    def _row_to_meta(row: "pd.Series") -> dict:
        meta: dict = {
            "start_time":     float(row["start_time"]),
            "end_time":       float(row["end_time"]),
            "recorder":       str(row.get("recorder", "")),
            "date":           str(row.get("date", "")),
            "time":           str(row.get("time", "")),
            "audio_duration": str(row.get("audio_duration", "")),
            "annotator":      str(row.get("annotator", "")),
        }
        for freq_col in ("low_frequency", "high_frequency"):
            if freq_col in row and pd.notna(row[freq_col]):
                meta[freq_col] = float(row[freq_col])
        if "bbox" in row and pd.notna(row["bbox"]):
            meta["bbox"] = row["bbox"]
        return meta


# ---------------------------------------------------------------------------
# Image loader
# ---------------------------------------------------------------------------

class BIRDeepImageLoader(BaseDatasetLoader):
    """Iterate over annotated spectrogram image patches in the BIRDeep dataset.

    Each annotation row in the CSV corresponds to one bird-call detection.
    The loader yields the full PNG spectrogram path together with a
    ``bbox_norm`` metadata key — a normalised YOLOv8 bounding box
    ``[x_center, y_center, w, h]`` — that image feature extractors use to
    crop the relevant patch before computing features.

    The cropping strategy mirrors how :class:`BIRDeepLoader` uses
    ``start_time`` / ``end_time`` on audio files: the full image is yielded
    and the extractor performs the crop, keeping the loader stateless with
    respect to image dimensions.

    Parameters
    ----------
    dataset_root:
        Path to the BIRDeep_AudioAnnotations directory.
    split:
        One of ``"train"``, ``"test"``, ``"validation"``, or ``"all"``.
    image_subdir:
        Sub-directory that contains the PNG spectrograms (default
        ``"images"``).
    include_augmented:
        Whether to include augmented recordings.
    min_bbox_area:
        Skip annotations whose normalised bounding box area (w × h) is
        smaller than this value.  Guards against degenerate detections.
    species_filter:
        If given, only annotations for the listed species are yielded.
    """

    def __init__(
        self,
        dataset_root:      Path | str,
        split:             str   = "train",
        image_subdir:      str   = "images",
        include_augmented: bool  = False,
        min_bbox_area:     float = 1e-5,
        species_filter:    Optional[set[str]] = None,
    ) -> None:
        if split not in _SPLIT_FILES:
            raise ValueError(
                f"split must be one of {list(_SPLIT_FILES)}, got {split!r}."
            )

        self.dataset_root      = Path(dataset_root)
        self.image_dir         = self.dataset_root / image_subdir
        self.split             = split
        self.include_augmented = include_augmented
        self.min_bbox_area     = min_bbox_area
        self.species_filter    = species_filter

        csv_path = self.dataset_root / _SPLIT_FILES[split]
        if not csv_path.exists():
            raise FileNotFoundError(
                f"CSV file not found: {csv_path}. "
                "Ensure dataset_root points to BIRDeep_AudioAnnotations/."
            )

        self._df = self._load_and_filter(csv_path)
        logger.info(
            "BIRDeepImageLoader [%s] – %d annotations across %d unique images.",
            split,
            len(self._df),
            self._df["path"].nunique(),
        )

    # ------------------------------------------------------------------
    # BaseDatasetLoader interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._df)

    def __iter__(self) -> Iterator[tuple[Path, Optional[str], dict]]:
        for _, row in self._df.iterrows():
            # Derive image path from audio path by swapping extension
            img_rel = Path(row["path"]).with_suffix(".PNG")
            img_path = self.image_dir / img_rel
            if not img_path.exists():
                logger.warning("Image not found, skipping: %s", img_path)
                continue

            label = str(row["specie"])
            meta  = self._row_to_meta(row)
            yield img_path, label, meta

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def species(self) -> list[str]:
        return sorted(self._df["specie"].unique().tolist())

    @property
    def n_species(self) -> int:
        return len(self.species)

    def species_counts(self) -> "pd.Series":
        return self._df["specie"].value_counts()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_and_filter(self, csv_path: Path) -> "pd.DataFrame":
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        # Only rows with a valid bbox are useful for the image loader
        mandatory = ["path", "specie", "bbox"]
        df = df.dropna(subset=mandatory)

        if not self.include_augmented:
            df = df[~df["path"].str.startswith("Data Augmentation")]

        if self.species_filter is not None:
            df = df[df["specie"].isin(self.species_filter)]

        return df.reset_index(drop=True)

    @staticmethod
    def _parse_bbox(raw: str) -> Optional[list[float]]:
        """Parse a CSV bbox string to ``[x_center, y_center, w, h]``.

        The CSV stores bboxes as Python list literals, e.g.:
        ``"[19, 0.213, 0.200, 0.011, 0.056]"``
        The first element is the class ID (discarded here — class identity is
        conveyed by the ``specie`` column).
        """
        import ast
        try:
            vals = ast.literal_eval(raw)
            # vals = [class_id, x_center, y_center, w, h]
            if len(vals) >= 5:
                return [float(v) for v in vals[1:5]]
        except Exception:
            pass
        return None

    def _row_to_meta(self, row: "pd.Series") -> dict:
        meta: dict = {
            "recorder":  str(row.get("recorder", "")),
            "date":      str(row.get("date", "")),
            "time":      str(row.get("time", "")),
            "annotator": str(row.get("annotator", "")),
        }
        for col in ("start_time", "end_time", "low_frequency", "high_frequency"):
            if col in row and pd.notna(row[col]):
                try:
                    meta[col] = float(row[col])
                except (ValueError, TypeError):
                    pass

        bbox_norm = self._parse_bbox(str(row.get("bbox", "")))
        if bbox_norm is not None:
            # Skip degenerate boxes
            if bbox_norm[2] * bbox_norm[3] >= self.min_bbox_area:
                meta["bbox_norm"] = bbox_norm

        return meta