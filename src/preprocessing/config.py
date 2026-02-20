"""
YAML-driven pipeline configuration.

Provides two dataclasses — :class:`ExperimentConfig` and
:class:`PipelineConfig` — and a :func:`load_config` helper that reads a YAML
file and returns a validated :class:`PipelineConfig`.

Single-run YAML example
-----------------------
.. code-block:: yaml

    # config/feature_extraction.yaml
    dataset:  data/raw/BIRDeep_AudioAnnotations
    split:    train
    extractor: audio_classical
    loader:    birdeep
    output:    data/processed/birdeep_classical_train

Multi-run batch YAML example
-----------------------------
.. code-block:: yaml

    # Top-level fields act as defaults for every experiment.
    dataset: data/raw/BIRDeep_AudioAnnotations
    split:   train

    experiments:
      - name:      birdeep_classical_train
        extractor: audio_classical
        loader:    birdeep

      - name:      birdeep_melspec_train
        extractor: audio_mel_spec
        loader:    birdeep

      - name:      birdeep_mobilenet_val
        extractor: image_mobilenet_v2
        loader:    birdeep_image
        split:     validation           # overrides top-level split

      - name:      video_classical_train
        extractor: video_classical
        loader:    video_folder
        dataset:   data/raw/my_videos  # overrides top-level dataset

Experiment-level fields always **override** top-level defaults.
The only required per-experiment fields are ``extractor`` and ``loader``
(which may also come from the top level).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Configuration for a single feature-extraction run.

    All fields may be *None* when constructed; :func:`load_config` fills
    them by merging top-level :class:`PipelineConfig` defaults before
    returning to the caller.

    Parameters
    ----------
    extractor:
        Registered extractor name (e.g. ``"audio_classical"``).
    loader:
        Registered loader name (e.g. ``"birdeep"``).
    name:
        Human-readable run name, used as the output directory name when
        *output* is absent.  Defaults to ``"<loader>_<extractor>_<split>"``.
    dataset:
        Path to the dataset root or file (inherited from
        :class:`PipelineConfig` when *None*).
    split:
        Dataset split: ``"train"``, ``"test"``, ``"validation"``,
        or ``"all"``.
    output:
        Output directory for the saved :class:`FeatureSet`.
        Defaults to ``data/processed/<name>``.
    max_samples:
        Cap the number of samples (useful for smoke-testing).
    label_col:
        Label column name — used by ``tabular`` and ``text_csv`` loaders.
    text_col:
        Text column name — used by the ``text_csv`` loader.
    audio_folder:
        Override dataset root for the ``audio_folder`` loader.
    image_folder:
        Override dataset root for the ``image_folder`` loader.
    text_folder:
        Override dataset root for the ``text_folder`` loader.
    video_folder:
        Override dataset root for the ``video_folder`` loader.
    """

    extractor:    str
    loader:       str
    name:         Optional[str] = None
    dataset:      Optional[str] = None
    split:        str           = "train"
    output:       Optional[str] = None
    max_samples:  Optional[int] = None
    label_col:    Optional[str] = None
    text_col:     str           = "text"
    audio_folder: Optional[str] = None
    image_folder: Optional[str] = None
    text_folder:  Optional[str] = None
    video_folder: Optional[str] = None

    def resolved_name(self) -> str:
        """Return *name* or auto-generate one from loader/extractor/split."""
        return self.name or f"{self.loader}_{self.extractor}_{self.split}"

    def resolved_output(self) -> str:
        """Return *output* or default to ``data/processed/<resolved_name>``."""
        return self.output or f"data/processed/{self.resolved_name()}"


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration.

    When *experiments* is non-empty the pipeline runs each experiment
    sequentially, using the top-level fields as defaults.

    When *experiments* is empty, the top-level *extractor* and *loader*
    fields are required and a single :class:`ExperimentConfig` is synthesised
    from the top-level fields.

    Parameters
    ----------
    dataset:
        Default dataset path for all experiments.
    split:
        Default split for all experiments.
    extractor:
        Default extractor name (required for single-run mode).
    loader:
        Default loader name (required for single-run mode).
    output:
        Default output directory.
    max_samples:
        Default sample cap.
    label_col:
        Default label column name.
    text_col:
        Default text column name.
    audio_folder:
        Default audio folder override.
    image_folder:
        Default image folder override.
    text_folder:
        Default text folder override.
    video_folder:
        Default video folder override.
    experiments:
        List of per-run configurations.  Empty → single-run mode.
    """

    dataset:      str           = "data/raw/BIRDeep_AudioAnnotations"
    split:        str           = "train"
    extractor:    Optional[str] = None
    loader:       Optional[str] = None
    output:       Optional[str] = None
    max_samples:  Optional[int] = None
    label_col:    Optional[str] = None
    text_col:     str           = "text"
    audio_folder: Optional[str] = None
    image_folder: Optional[str] = None
    text_folder:  Optional[str] = None
    video_folder: Optional[str] = None
    experiments:  list[ExperimentConfig] = field(default_factory=list)

    def resolved_experiments(self) -> list[ExperimentConfig]:
        """Return resolved experiments, merging top-level defaults.

        In single-run mode (no experiments list) a single
        :class:`ExperimentConfig` is synthesised from the top-level fields.

        Raises
        ------
        ValueError
            If an experiment (or the top-level config in single-run mode)
            is missing a required ``extractor`` or ``loader``.
        """
        if not self.experiments:
            # Single-run mode: require extractor + loader at top level
            if not self.extractor or not self.loader:
                raise ValueError(
                    "PipelineConfig: 'extractor' and 'loader' are required "
                    "when no 'experiments' list is provided."
                )
            return [
                ExperimentConfig(
                    extractor=self.extractor,
                    loader=self.loader,
                    dataset=self.dataset,
                    split=self.split,
                    output=self.output,
                    max_samples=self.max_samples,
                    label_col=self.label_col,
                    text_col=self.text_col,
                    audio_folder=self.audio_folder,
                    image_folder=self.image_folder,
                    text_folder=self.text_folder,
                    video_folder=self.video_folder,
                )
            ]

        resolved: list[ExperimentConfig] = []
        for i, exp in enumerate(self.experiments):
            # Fill missing experiment fields from top-level defaults
            merged = ExperimentConfig(
                extractor=exp.extractor or self.extractor or "",
                loader=exp.loader or self.loader or "",
                name=exp.name,
                dataset=exp.dataset or self.dataset,
                split=exp.split or self.split,
                output=exp.output or self.output,
                max_samples=exp.max_samples if exp.max_samples is not None else self.max_samples,
                label_col=exp.label_col or self.label_col,
                text_col=exp.text_col or self.text_col,
                audio_folder=exp.audio_folder or self.audio_folder,
                image_folder=exp.image_folder or self.image_folder,
                text_folder=exp.text_folder or self.text_folder,
                video_folder=exp.video_folder or self.video_folder,
            )
            if not merged.extractor:
                raise ValueError(
                    f"Experiment #{i} is missing 'extractor'. "
                    "Set it in the experiment or at the top level."
                )
            if not merged.loader:
                raise ValueError(
                    f"Experiment #{i} is missing 'loader'. "
                    "Set it in the experiment or at the top level."
                )
            resolved.append(merged)
        return resolved


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def load_config(path: Path | str) -> PipelineConfig:
    """Parse a YAML config file and return a :class:`PipelineConfig`.

    Parameters
    ----------
    path:
        Path to the YAML file.

    Returns
    -------
    PipelineConfig
        Populated config with all experiments validated.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the YAML is structurally invalid or experiments are missing
        required fields.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw: dict = yaml.safe_load(path.read_text()) or {}

    # Extract experiments list separately
    raw_experiments: list[dict] = raw.pop("experiments", []) or []

    # Build top-level config (ignoring unknown keys to be forward-compatible)
    _top_keys = {f for f in PipelineConfig.__dataclass_fields__}
    top_kwargs = {k: v for k, v in raw.items() if k in _top_keys}
    cfg = PipelineConfig(**top_kwargs)

    # Parse each experiment
    _exp_keys = {f for f in ExperimentConfig.__dataclass_fields__}
    for raw_exp in raw_experiments:
        exp_kwargs = {k: v for k, v in raw_exp.items() if k in _exp_keys}
        # extractor and loader may be absent (inherited from top level)
        exp = ExperimentConfig(
            extractor=exp_kwargs.pop("extractor", ""),
            loader=exp_kwargs.pop("loader", ""),
            **exp_kwargs,
        )
        cfg.experiments.append(exp)

    # Validate by resolving (raises ValueError on missing required fields)
    cfg.resolved_experiments()

    return cfg