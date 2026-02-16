"""
BIRDeep Audio Annotations dataset loader.

Reads the pre-split CSV files that ship with the dataset
(``train_file.csv``, ``test_file.csv``, ``validation_file.csv``,
``dataset.csv``) and yields one entry per annotated bird-call segment:

    (audio_path, species_label, metadata_dict)

Each annotation describes a single bird vocalisation detected within a
one-minute WAV recording.  The metadata dict carries all per-annotation
fields so that feature extractors can use segment boundaries, frequency
range, recorder site, etc.

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