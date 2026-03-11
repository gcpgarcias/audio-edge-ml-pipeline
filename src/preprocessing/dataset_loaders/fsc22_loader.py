"""
FSC22 (Field Sound Classification 2022) dataset loader.

Dataset layout (as distributed):
    <dataset_root>/
        Audio Wise V1.0-*/
            Audio Wise V1.0/
                <ClassID>_<FileID>.wav   (2025 files, all in one flat directory)
        Metadata-*/
            Metadata/
                Metadata V1.0 FSC22.csv  (Source File Name, Dataset File Name,
                                          Class ID, Class Name)

Since FSC22 ships without pre-defined train/val/test splits, this loader
performs a **deterministic stratified split** at construction time using a
fixed random seed.  The default 70 / 15 / 15 ratio gives balanced splits
across all 27 classes.

Each sample yields::

    (audio_path, class_name, {
        "filename":   str,
        "class_id":   int,
        "class_name": str,
        "split":      str,   # "train" | "validation" | "test"
    })
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from ..feature_extraction.base import BaseDatasetLoader

logger = logging.getLogger(__name__)

_VALID_SPLITS = ("train", "validation", "test", "all")


def _find_path(root: Path, glob_pattern: str) -> Optional[Path]:
    """Return the first match for *glob_pattern* under *root*, or None."""
    matches = list(root.glob(glob_pattern))
    return matches[0] if matches else None


class FSC22Loader(BaseDatasetLoader):
    """Load audio clips from the FSC22 flat-directory dataset.

    FSC22 ships all audio files in a single directory with a companion CSV
    that maps filenames to class labels.  This loader reads the CSV, resolves
    file paths, and applies a deterministic stratified split so that
    downstream pipeline stages can request the ``"train"``, ``"validation"``,
    or ``"test"`` partition.

    Parameters
    ----------
    dataset_root:
        Path to the top-level FSC22 directory (the one that contains the
        ``Audio Wise V1.0-*`` and ``Metadata-*`` sub-directories).
    split:
        Partition to return.  One of ``"train"``, ``"validation"``,
        ``"test"``, or ``"all"``.
    class_filter:
        If given, only clips whose ``Class Name`` is in this collection
        are yielded.
    train_ratio:
        Fraction of each class assigned to the training split (default 0.70).
    val_ratio:
        Fraction of each class assigned to the validation split (default
        0.15).  The remainder becomes the test split.
    seed:
        Random seed used for the stratified split (default 42).
    """

    def __init__(
        self,
        dataset_root:   Path | str,
        split:          str                 = "train",
        class_filter:   Optional[set[str]]  = None,
        train_ratio:    float               = 0.70,
        val_ratio:      float               = 0.15,
        seed:           int                 = 42,
    ) -> None:
        if split not in _VALID_SPLITS:
            raise ValueError(
                f"split must be one of {list(_VALID_SPLITS)}, got {split!r}."
            )

        self.dataset_root = Path(dataset_root)
        self.split        = split
        self.class_filter = class_filter

        audio_dir = _find_path(self.dataset_root, "Audio Wise V1.0-*/Audio Wise V1.0")
        if audio_dir is None or not audio_dir.is_dir():
            raise FileNotFoundError(
                f"Could not find 'Audio Wise V1.0' directory under {self.dataset_root}. "
                "Expected layout: <root>/Audio Wise V1.0-*/<Audio Wise V1.0>/."
            )

        csv_path = _find_path(self.dataset_root, "Metadata-*/Metadata/*.csv")
        if csv_path is None:
            raise FileNotFoundError(
                f"Could not find FSC22 metadata CSV under {self.dataset_root}. "
                "Expected layout: <root>/Metadata-*/Metadata/Metadata V1.0 FSC22.csv."
            )

        self._audio_dir = audio_dir
        self._df = self._load_and_split(csv_path, train_ratio, val_ratio, seed)

        logger.info(
            "FSC22Loader [%s] – %d clips across %d classes.",
            split,
            len(self._df),
            self._df["Class Name"].nunique(),
        )

    # ------------------------------------------------------------------
    # BaseDatasetLoader interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._df)

    def __iter__(self) -> Iterator[tuple[Path, Optional[str], dict]]:
        for _, row in self._df.iterrows():
            audio_path = self._audio_dir / row["Dataset File Name"]
            if not audio_path.exists():
                logger.warning("Audio file not found, skipping: %s", audio_path)
                continue
            label = str(row["Class Name"])
            meta  = {
                "filename":   row["Dataset File Name"],
                "class_id":   int(row["Class ID"]),
                "class_name": label,
                "split":      row["_split"],
            }
            yield audio_path, label, meta

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def class_names(self) -> list[str]:
        """Sorted list of class names present in this split."""
        return sorted(self._df["Class Name"].unique().tolist())

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    def class_counts(self) -> "pd.Series":
        """Return sample counts per class for this split, descending."""
        return self._df["Class Name"].value_counts()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_and_split(
        self,
        csv_path:    Path,
        train_ratio: float,
        val_ratio:   float,
        seed:        int,
    ) -> "pd.DataFrame":
        from sklearn.model_selection import train_test_split

        df = pd.read_csv(csv_path, on_bad_lines="warn")
        df.columns = df.columns.str.strip()

        # Drop rows missing mandatory fields
        mandatory = ["Dataset File Name", "Class ID", "Class Name"]
        before = len(df)
        df = df.dropna(subset=mandatory)
        if len(df) < before:
            logger.warning("Dropped %d rows with missing fields.", before - len(df))

        df["Class Name"] = df["Class Name"].str.strip()

        # Optional class filter
        if self.class_filter is not None:
            df = df[df["Class Name"].isin(self.class_filter)]

        if df.empty:
            logger.warning("FSC22Loader: no samples remain after filtering.")
            df["_split"] = []
            return df.reset_index(drop=True)

        # Deterministic stratified split: train / (val + test)
        test_ratio = 1.0 - train_ratio - val_ratio
        if test_ratio < 0:
            raise ValueError(
                f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) > 1.0"
            )

        labels = df["Class Name"].tolist()

        train_idx, remaining_idx = train_test_split(
            range(len(df)),
            test_size=1.0 - train_ratio,
            stratify=labels,
            random_state=seed,
        )

        remaining_labels = [labels[i] for i in remaining_idx]
        val_fraction = val_ratio / (val_ratio + test_ratio) if (val_ratio + test_ratio) > 0 else 0.5
        val_idx, test_idx = train_test_split(
            remaining_idx,
            test_size=1.0 - val_fraction,
            stratify=remaining_labels,
            random_state=seed,
        )

        split_map = (
            {i: "train"      for i in train_idx}
            | {i: "validation" for i in val_idx}
            | {i: "test"       for i in test_idx}
        )
        df["_split"] = [split_map[i] for i in range(len(df))]

        if self.split == "all":
            result = df
        else:
            result = df[df["_split"] == self.split]

        return result.reset_index(drop=True)
