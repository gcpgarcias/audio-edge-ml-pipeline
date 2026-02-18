"""
Feature extraction pipeline orchestrator.

Ties a :class:`BaseDatasetLoader` to a :class:`BaseFeatureExtractor`,
runs the extraction, and handles persistence of the resulting
:class:`FeatureSet`.

Typical usage
-------------
::

    from pathlib import Path
    from src.preprocessing.dataset_loaders import BIRDeepLoader
    from src.preprocessing.feature_extraction import get
    from src.preprocessing.pipeline import FeaturePipeline

    loader    = BIRDeepLoader("data/raw/BIRDeep_AudioAnnotations", split="train")
    extractor = get("audio_classical")()          # instantiate with defaults
    pipeline  = FeaturePipeline(loader, extractor)

    fs = pipeline.run()                           # returns a FeatureSet
    print(fs)
    # FeatureSet(modality='audio', feature_type='classical',
    #            n_samples=2877, feature_shape=(302,), labels=36 classes)

    pipeline.save(fs, Path("data/processed/birdeep_classical"))
    fs2 = FeaturePipeline.load(Path("data/processed/birdeep_classical"))

Persistence layout
------------------
::

    <output_dir>/
        features.npy         float32 array, shape (N, *feature_dims)
        labels.npy           int32 array, shape (N,)  [absent if unsupervised]
        label_names.json     list[str]                [absent if unsupervised]
        metadata.json        list[dict]
        info.json            {feature_type, modality, n_samples, feature_shape}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .feature_extraction.base import BaseFeatureExtractor, BaseDatasetLoader, FeatureSet

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Orchestrates feature extraction and manages FeatureSet persistence.

    Parameters
    ----------
    loader:
        Any :class:`BaseDatasetLoader` that yields ``(path, label, meta)``
        tuples.
    extractor:
        Any :class:`BaseFeatureExtractor` compatible with the loader's
        modality.
    """

    def __init__(
        self,
        loader:    BaseDatasetLoader,
        extractor: BaseFeatureExtractor,
    ) -> None:
        self.loader    = loader
        self.extractor = extractor

    # ------------------------------------------------------------------
    # Running the pipeline
    # ------------------------------------------------------------------

    def run(self, max_samples: Optional[int] = None) -> FeatureSet:
        """Extract features for all (or *max_samples*) samples.

        Parameters
        ----------
        max_samples:
            Cap the number of samples processed.  Useful for quick
            smoke-tests without touching the full dataset.

        Returns
        -------
        FeatureSet
        """
        logger.info(
            "Starting extraction: loader=%s (%d samples), extractor=%s",
            type(self.loader).__name__,
            len(self.loader),
            self.extractor.name,
        )
        fs = self.extractor.extract_dataset(self.loader, max_samples=max_samples)
        logger.info("Extraction complete: %s", fs)
        return fs

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save(fs: FeatureSet, output_dir: Path | str) -> None:
        """Persist *fs* to *output_dir*.

        Parameters
        ----------
        fs:
            The :class:`FeatureSet` to save.
        output_dir:
            Directory that will be created if it does not exist.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Features array (always present)
        np.save(output_dir / "features.npy", fs.features)

        # Labels (supervised / semi-supervised only)
        if fs.labels is not None:
            np.save(output_dir / "labels.npy", fs.labels)
        if fs.label_names is not None:
            (output_dir / "label_names.json").write_text(
                json.dumps(fs.label_names, indent=2)
            )

        # Cluster assignments (post-unsupervised-fit)
        if fs.cluster_assignments is not None:
            np.save(output_dir / "cluster_assignments.npy", fs.cluster_assignments)

        # Per-sample metadata
        (output_dir / "metadata.json").write_text(
            json.dumps(fs.metadata, indent=2, default=str)
        )

        # Human-readable manifest
        info = {
            "feature_type":  fs.feature_type,
            "modality":      fs.modality,
            "n_samples":     fs.n_samples,
            "feature_shape": list(fs.feature_shape),
            "n_classes":     fs.n_classes,
            "is_supervised": fs.is_supervised,
        }
        (output_dir / "info.json").write_text(json.dumps(info, indent=2))

        logger.info("FeatureSet saved to %s", output_dir)

    @staticmethod
    def load(output_dir: Path | str) -> FeatureSet:
        """Reload a :class:`FeatureSet` previously saved by :meth:`save`.

        Parameters
        ----------
        output_dir:
            Directory written by :meth:`save`.

        Returns
        -------
        FeatureSet

        Raises
        ------
        FileNotFoundError
            If *features.npy* or *info.json* are missing.
        """
        output_dir = Path(output_dir)

        features_path = output_dir / "features.npy"
        info_path     = output_dir / "info.json"
        for p in (features_path, info_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Expected file not found: {p}. "
                    "Was this directory written by FeaturePipeline.save()?"
                )

        features = np.load(features_path)
        info     = json.loads(info_path.read_text())

        labels = (
            np.load(output_dir / "labels.npy")
            if (output_dir / "labels.npy").exists()
            else None
        )
        label_names = (
            json.loads((output_dir / "label_names.json").read_text())
            if (output_dir / "label_names.json").exists()
            else None
        )
        cluster_assignments = (
            np.load(output_dir / "cluster_assignments.npy")
            if (output_dir / "cluster_assignments.npy").exists()
            else None
        )
        metadata = (
            json.loads((output_dir / "metadata.json").read_text())
            if (output_dir / "metadata.json").exists()
            else []
        )

        fs = FeatureSet(
            features=features,
            feature_type=info["feature_type"],
            modality=info["modality"],
            metadata=metadata,
            labels=labels,
            label_names=label_names,
            cluster_assignments=cluster_assignments,
        )
        logger.info("FeatureSet loaded from %s: %s", output_dir, fs)
        return fs


# ---------------------------------------------------------------------------
# Quick-start CLI
# ---------------------------------------------------------------------------

def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(
        description="Run the feature extraction pipeline on a dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset",
        default="data/raw/BIRDeep_AudioAnnotations",
        help="Path to the dataset root directory.",
    )
    p.add_argument(
        "--loader",
        default="birdeep",
        choices=["birdeep", "birdeep_image", "image_folder"],
        help="Dataset loader to use.",
    )
    p.add_argument(
        "--image-folder",
        default=None,
        help="Root path for image_folder loader.",
    )
    p.add_argument(
        "--split",
        default="train",
        choices=["train", "test", "validation", "all"],
    )
    p.add_argument(
        "--extractor",
        default="audio_classical",
        help="Registered extractor name (see feature_extraction.list_extractors()).",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output directory for the saved FeatureSet. "
             "Defaults to data/processed/<loader>_<extractor>_<split>.",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit the number of samples (for quick testing).",
    )
    return p


def main() -> None:
    from src.preprocessing.dataset_loaders import (
        BIRDeepImageLoader,
        BIRDeepLoader,
        ImageFolderLoader,
    )
    from src.preprocessing.feature_extraction import get  # noqa: F401 – triggers registration

    args = _build_arg_parser().parse_args()

    if args.loader == "birdeep":
        loader = BIRDeepLoader(args.dataset, split=args.split)
    elif args.loader == "birdeep_image":
        loader = BIRDeepImageLoader(args.dataset, split=args.split)
    elif args.loader == "image_folder":
        root = args.image_folder or args.dataset
        loader = ImageFolderLoader(root, split=args.split)
    else:
        raise ValueError(f"Unknown loader: {args.loader!r}")

    extractor = get(args.extractor)()

    output_dir = Path(
        args.output
        or f"data/processed/{args.loader}_{args.extractor}_{args.split}"
    )

    pipeline = FeaturePipeline(loader, extractor)
    fs = pipeline.run(max_samples=args.max_samples)
    FeaturePipeline.save(fs, output_dir)
    print(f"Saved: {fs}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()