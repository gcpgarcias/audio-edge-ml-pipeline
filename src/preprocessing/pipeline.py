"""
Feature extraction pipeline orchestrator.

Ties a :class:`BaseDatasetLoader` to a :class:`BaseFeatureExtractor`,
runs the extraction, and handles persistence of the resulting
:class:`FeatureSet`.

Typical programmatic usage
--------------------------
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
        cluster_assignments.npy int32 (N,)            [absent if not clustered]
        metadata.json        list[dict]
        info.json            {feature_type, modality, n_samples, feature_shape}

CLI usage — single run (flags)
-------------------------------
::

    python -m src.preprocessing.pipeline \\
        --loader birdeep --dataset data/raw/BIRDeep_AudioAnnotations \\
        --split train --extractor audio_classical \\
        --output data/processed/birdeep_classical

CLI usage — config-driven (single or multi-run)
-----------------------------------------------
::

    python -m src.preprocessing.pipeline --config config/feature_extraction.yaml

See :mod:`src.preprocessing.config` for the YAML schema.
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
# Loader factory (shared by flag-based and config-based paths)
# ---------------------------------------------------------------------------

def _build_loader(
    loader_name:  str,
    dataset:      str,
    split:        str,
    label_col:    Optional[str] = None,
    text_col:     str           = "text",
    audio_folder: Optional[str] = None,
    image_folder: Optional[str] = None,
    text_folder:  Optional[str] = None,
    video_folder: Optional[str] = None,
) -> BaseDatasetLoader:
    """Instantiate the requested loader with the given parameters.

    Parameters
    ----------
    loader_name:
        One of the recognised loader keys (e.g. ``"birdeep"``,
        ``"audio_folder"``, ``"video_folder"``).
    dataset:
        Default dataset root / file path.
    split:
        Dataset split (``"train"``, ``"test"``, ``"validation"``, ``"all"``).
    label_col, text_col, audio_folder, image_folder, text_folder, video_folder:
        Loader-specific overrides (see CLI args for semantics).

    Raises
    ------
    ValueError
        If *loader_name* is not recognised.
    """
    from src.preprocessing.dataset_loaders import (
        AudioFolderLoader,
        BIRDeepImageLoader,
        BIRDeepLoader,
        ImageFolderLoader,
        TabularLoader,
        TextCSVLoader,
        TextFolderLoader,
        TextJSONLoader,
        VideoFolderLoader,
    )

    if loader_name == "birdeep":
        return BIRDeepLoader(dataset, split=split)
    elif loader_name == "birdeep_image":
        return BIRDeepImageLoader(dataset, split=split)
    elif loader_name == "audio_folder":
        root = audio_folder or dataset
        return AudioFolderLoader(root, split=split)
    elif loader_name == "image_folder":
        root = image_folder or dataset
        return ImageFolderLoader(root, split=split)
    elif loader_name == "text_folder":
        root = text_folder or dataset
        return TextFolderLoader(root, split=split)
    elif loader_name == "text_json":
        return TextJSONLoader(dataset)
    elif loader_name == "text_csv":
        return TextCSVLoader(dataset, text_col=text_col, label_col=label_col)
    elif loader_name == "tabular":
        return TabularLoader(dataset, label_col=label_col)
    elif loader_name == "video_folder":
        root = video_folder or dataset
        return VideoFolderLoader(root, split=split)
    else:
        raise ValueError(
            f"Unknown loader: {loader_name!r}. "
            "Valid choices: birdeep, birdeep_image, audio_folder, image_folder, "
            "text_folder, text_json, text_csv, tabular, video_folder."
        )


# ---------------------------------------------------------------------------
# Quick-start CLI
# ---------------------------------------------------------------------------

_LOADER_CHOICES = [
    "birdeep", "birdeep_image",
    "audio_folder", "image_folder", "video_folder",
    "text_folder", "text_json", "text_csv",
    "tabular",
]


def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "Run the feature extraction pipeline on a dataset.\n\n"
            "Two modes:\n"
            "  1. Flag-based (single run): supply --loader, --extractor, etc.\n"
            "  2. Config-driven (single or batch): supply --config <yaml>.\n"
            "     --config and the flag-based args are mutually exclusive."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- Config-file mode -------------------------------------------
    p.add_argument(
        "--config",
        default=None,
        metavar="YAML",
        help=(
            "Path to a YAML pipeline config file.  When supplied, all other "
            "flags are ignored (settings come from the file)."
        ),
    )

    # ---- Flag-based mode --------------------------------------------
    p.add_argument(
        "--dataset",
        default="data/raw/BIRDeep_AudioAnnotations",
        help="Path to the dataset root directory or file.",
    )
    p.add_argument(
        "--loader",
        default="birdeep",
        choices=_LOADER_CHOICES,
        help="Dataset loader to use.",
    )
    p.add_argument(
        "--audio-folder",
        default=None,
        help="Root path for audio_folder loader (overrides --dataset).",
    )
    p.add_argument(
        "--image-folder",
        default=None,
        help="Root path for image_folder loader (overrides --dataset).",
    )
    p.add_argument(
        "--text-folder",
        default=None,
        help="Root path for text_folder loader (overrides --dataset).",
    )
    p.add_argument(
        "--video-folder",
        default=None,
        help="Root path for video_folder loader (overrides --dataset).",
    )
    p.add_argument(
        "--label-col",
        default=None,
        help="Label column name (text_csv and tabular loaders).",
    )
    p.add_argument(
        "--text-col",
        default="text",
        help="Text column name (text_csv loader).",
    )
    p.add_argument(
        "--split",
        default="train",
        choices=["train", "test", "validation", "all"],
    )
    p.add_argument(
        "--extractor",
        default="audio_classical",
        help=(
            "Registered extractor name. "
            "Run python -c \"from src.preprocessing.feature_extraction import "
            "list_extractors; print(list_extractors())\" to see all options."
        ),
    )
    p.add_argument(
        "--output",
        default=None,
        help=(
            "Output directory for the saved FeatureSet. "
            "Defaults to data/processed/<loader>_<extractor>_<split>."
        ),
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit the number of samples (for quick testing).",
    )
    return p


def _run_experiment(exp) -> None:
    """Execute a single :class:`ExperimentConfig` end-to-end."""
    from src.preprocessing.feature_extraction import get  # triggers registration

    loader = _build_loader(
        loader_name=exp.loader,
        dataset=exp.dataset or "data/raw/BIRDeep_AudioAnnotations",
        split=exp.split,
        label_col=exp.label_col,
        text_col=exp.text_col,
        audio_folder=exp.audio_folder,
        image_folder=exp.image_folder,
        text_folder=exp.text_folder,
        video_folder=exp.video_folder,
    )
    extractor  = get(exp.extractor)()
    output_dir = Path(exp.resolved_output())
    pipeline   = FeaturePipeline(loader, extractor)
    fs         = pipeline.run(max_samples=exp.max_samples)
    FeaturePipeline.save(fs, output_dir)
    print(f"[{exp.resolved_name()}] {fs}")
    print(f"  -> {output_dir}")


def main() -> None:
    args = _build_arg_parser().parse_args()

    if args.config:
        # ------ Config-driven mode -----------------------------------
        from .config import load_config

        cfg         = load_config(args.config)
        experiments = cfg.resolved_experiments()
        print(f"Config: {args.config}  ({len(experiments)} experiment(s))")

        for exp in experiments:
            print(f"\nRunning: {exp.resolved_name()} ...")
            _run_experiment(exp)

        print("\nAll experiments complete.")

    else:
        # ------ Flag-based mode --------------------------------------
        from .config import ExperimentConfig

        exp = ExperimentConfig(
            extractor=args.extractor,
            loader=args.loader,
            dataset=args.dataset,
            split=args.split,
            output=args.output,
            max_samples=args.max_samples,
            label_col=args.label_col,
            text_col=args.text_col,
            audio_folder=args.audio_folder,
            image_folder=args.image_folder,
            text_folder=args.text_folder,
            video_folder=args.video_folder,
        )
        _run_experiment(exp)


if __name__ == "__main__":
    main()