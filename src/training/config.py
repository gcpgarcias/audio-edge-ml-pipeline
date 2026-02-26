"""
Configuration dataclasses and YAML loader for Stage 3 (model training).

Usage
-----
Load a training config YAML:

    cfg = load_train_config(Path("config/training.yaml"))
    for run in cfg.resolved_runs():
        trainer = get_model(run.model)(**run.params)

Schema
------
::

    # Top-level defaults (inherited by every run)
    features_dir:      data/processed/birdeep_classical_train
    output_dir:        data/models
    experiment:        birdeep-classification
    mlflow_uri:        null          # null → env var MLFLOW_TRACKING_URI or "mlflow/"
    val_split:         0.2
    features_test_dir: null          # optional held-out test FeatureSet

    runs:
      - model:             svm
        name:              birdeep_svm_rbf    # optional; defaults to "<model>_<timestamp>"
        features_dir:      ...               # overrides top-level
        features_test_dir: ...               # overrides top-level
        output_dir:        ...               # overrides top-level
        val_split:         0.15              # overrides top-level
        params:
          C: 10.0
          kernel: rbf

      - model: cnn
        params:
          filters:  32
          n_blocks: 3
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
class ModelRunConfig:
    """Configuration for a single model training run.

    Parameters
    ----------
    model:
        Registered trainer name (e.g. ``"svm"``, ``"cnn"``).
    name:
        Human-readable run name used for MLflow and output directory naming.
        Defaults to ``"<model>"`` if not provided.
    features_dir:
        Path to a Stage 2 ``FeatureSet`` directory.  Overrides the top-level
        ``features_dir`` when set.
    features_test_dir:
        Optional path to a held-out test ``FeatureSet``.  Overrides
        top-level ``features_test_dir`` when set.
    output_dir:
        Output directory for model artefacts.  Overrides top-level
        ``output_dir`` when set.
    val_split:
        Fraction of training data to use for validation (0 < val_split < 1).
    params:
        Keyword arguments forwarded to the trainer constructor.
    """

    model:             str
    name:              Optional[str]  = None
    features_dir:      Optional[str]  = None
    features_test_dir: Optional[str]  = None
    output_dir:        Optional[str]  = None
    val_split:         float          = 0.2
    params:            dict           = field(default_factory=dict)


@dataclass
class TrainConfig:
    """Top-level training configuration.

    Parameters
    ----------
    features_dir:
        Default path to a Stage 2 ``FeatureSet`` directory.
    output_dir:
        Default root directory for model artefacts.
    experiment:
        MLflow experiment name.
    mlflow_uri:
        MLflow tracking URI.  *None* → use ``MLFLOW_TRACKING_URI`` env var,
        falling back to ``"mlflow/"`` (local file store).
    val_split:
        Default validation fraction.
    features_test_dir:
        Optional default path to a held-out test ``FeatureSet``.
    runs:
        Per-model run configurations.  Each can override any top-level field.
    """

    features_dir:             str
    output_dir:               str
    experiment:               str                  = "ml-pipeline"
    mlflow_uri:               Optional[str]        = None
    val_split:                float                = 0.2
    features_test_dir:        Optional[str]        = None
    runs:                     list[ModelRunConfig] = field(default_factory=list)
    # Auto-selection written to <output_dir>/shortlist.json at end of sweep
    auto_select:              bool                 = True
    auto_select_top_n:        int                  = 5
    auto_select_metric:       str                  = "val_f1_macro"
    auto_select_min_accuracy: Optional[float]      = None

    def resolved_runs(self) -> list[ModelRunConfig]:
        """Return a list of :class:`ModelRunConfig` objects with top-level
        defaults filled in where individual runs omit them.
        """
        resolved = []
        for run in self.runs:
            resolved.append(
                ModelRunConfig(
                    model             = run.model,
                    name              = run.name or run.model,
                    features_dir      = run.features_dir      or self.features_dir,
                    features_test_dir = run.features_test_dir or self.features_test_dir,
                    output_dir        = run.output_dir        or self.output_dir,
                    val_split         = run.val_split         if run.val_split != 0.2
                                        else self.val_split,
                    params            = run.params,
                )
            )
        return resolved


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def load_train_config(path: Path) -> TrainConfig:
    """Parse a training YAML file and return a :class:`TrainConfig`.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If required fields (``features_dir``, ``output_dir``) are missing and
        cannot be inferred from per-run overrides.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Training config not found: {path}")

    with path.open() as fh:
        raw = yaml.safe_load(fh) or {}

    # Top-level defaults
    features_dir             = raw.get("features_dir", "")
    output_dir               = raw.get("output_dir", "data/models")
    experiment               = raw.get("experiment", "ml-pipeline")
    mlflow_uri               = raw.get("mlflow_uri", None)
    val_split                = float(raw.get("val_split", 0.2))
    features_test_dir        = raw.get("features_test_dir", None)
    auto_select              = bool(raw.get("auto_select", True))
    auto_select_top_n        = int(raw.get("auto_select_top_n", 5))
    auto_select_metric       = str(raw.get("auto_select_metric", "val_f1_macro"))
    auto_select_min_accuracy = raw.get("auto_select_min_accuracy", None)
    if auto_select_min_accuracy is not None:
        auto_select_min_accuracy = float(auto_select_min_accuracy)

    if not features_dir:
        raise ValueError(
            "TrainConfig requires 'features_dir' at the top level "
            "(or every run must specify its own 'features_dir')."
        )

    # Parse runs
    runs: list[ModelRunConfig] = []
    for item in raw.get("runs", []):
        if "model" not in item:
            raise ValueError(f"Each run must specify a 'model' key. Got: {item}")
        runs.append(
            ModelRunConfig(
                model             = item["model"],
                name              = item.get("name"),
                features_dir      = item.get("features_dir"),
                features_test_dir = item.get("features_test_dir"),
                output_dir        = item.get("output_dir"),
                val_split         = float(item.get("val_split", 0.2)),
                params            = item.get("params") or {},
            )
        )

    return TrainConfig(
        features_dir             = features_dir,
        output_dir               = output_dir,
        experiment               = experiment,
        mlflow_uri               = mlflow_uri,
        val_split                = val_split,
        features_test_dir        = features_test_dir,
        runs                     = runs,
        auto_select              = auto_select,
        auto_select_top_n        = auto_select_top_n,
        auto_select_metric       = auto_select_metric,
        auto_select_min_accuracy = auto_select_min_accuracy,
    )