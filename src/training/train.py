"""
Stage 3 — Model Training entry point.

Two operating modes
-------------------

**Flag-based (single run)**::

    python -m src.training.train \\
        --features data/processed/birdeep_classical_train \\
        --model svm \\
        --output data/models/birdeep_svm \\
        [--features-test data/processed/birdeep_classical_test] \\
        [--val-split 0.2] \\
        [--experiment birdeep-classification] \\
        [--max-samples 500] \\
        [--param C=10.0] [--param kernel=linear]

**Config-driven multi-model sweep**::

    python -m src.training.train --config config/training.yaml

After all runs complete, a ``shortlist.json`` is automatically written to the
output directory via Stage 5 pre-optimisation selection.  Pass
``--no-auto-select`` to suppress this.

Each run:
1. Loads a :class:`~src.preprocessing.pipeline.FeatureSet` from disk.
2. Splits into train / val (stratified where possible).
3. Instantiates the requested trainer.
4. Calls ``trainer.fit()``.
5. Optionally evaluates on a held-out test ``FeatureSet``.
6. Logs everything to MLflow (local file store by default).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# ── project imports ──────────────────────────────────────────────────────────
# Ensure package root is on sys.path when invoked as __main__
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.preprocessing.pipeline import FeaturePipeline
from src.training.models import get_model, list_models
from src.training.config import TrainConfig, ModelRunConfig, load_train_config
from src.training import evaluate as ev

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MLflow helpers
# ---------------------------------------------------------------------------

def _setup_mlflow(uri: Optional[str], experiment: str):
    """Configure MLflow tracking URI and experiment, return mlflow module."""
    import mlflow

    tracking_uri = uri or os.getenv("MLFLOW_TRACKING_URI", "mlruns/")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    logger.info("MLflow tracking URI: %s  experiment: %s", tracking_uri, experiment)
    return mlflow


# ---------------------------------------------------------------------------
# Core per-run logic
# ---------------------------------------------------------------------------

def _run_one(
    run:        ModelRunConfig,
    experiment: str,
    mlflow_uri: Optional[str],
    max_samples: Optional[int] = None,
) -> None:
    """Execute a single training run described by *run*."""

    import mlflow

    # ── 1. Load FeatureSet ────────────────────────────────────────────────
    features_dir = Path(run.features_dir)
    logger.info("[%s] Loading features from %s", run.name, features_dir)
    feature_set = FeaturePipeline.load(features_dir)

    X: np.ndarray = feature_set.features
    y: np.ndarray = feature_set.labels
    label_names: list[str] = feature_set.label_names or []

    if y is None:
        raise ValueError(
            f"FeatureSet at '{features_dir}' has no labels. "
            "Supervised training requires labelled data."
        )

    if max_samples and max_samples < len(X):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), max_samples, replace=False)
        X, y = X[idx], y[idx]
        logger.info("[%s] Subsampled to %d samples", run.name, max_samples)

    # ── 2. Train / val split ──────────────────────────────────────────────
    from sklearn.model_selection import train_test_split

    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=run.val_split, random_state=42, stratify=y
        )
    except ValueError:
        # Fallback if a class has only 1 sample (can't stratify)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=run.val_split, random_state=42
        )

    logger.info(
        "[%s] Train: %d  Val: %d  Classes: %d",
        run.name, len(X_train), len(X_val), len(label_names),
    )

    # ── 3. Output directory ───────────────────────────────────────────────
    output_dir = Path(run.output_dir) / run.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 4. MLflow run ─────────────────────────────────────────────────────
    mlflow_module = _setup_mlflow(mlflow_uri, experiment)
    run_name = f"{run.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow_module.start_run(run_name=run_name) as active_run:

        # ── 5. Train ──────────────────────────────────────────────────────
        trainer_cls = get_model(run.model)
        trainer = trainer_cls(**run.params)

        result = trainer.fit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            label_names=label_names,
            run_name=run_name,
            output_dir=output_dir,
            mlflow_run=active_run,
        )

        # ── 6. Optional test-set evaluation ───────────────────────────────
        if run.features_test_dir:
            test_dir = Path(run.features_test_dir)
            logger.info("[%s] Evaluating on test set: %s", run.name, test_dir)
            try:
                test_fs = FeaturePipeline.load(test_dir)
                X_test, y_test = test_fs.features, test_fs.labels

                if y_test is not None:
                    y_pred_test = trainer.predict(X_test)
                    y_proba_test = trainer.predict_proba(X_test)
                    test_metrics = ev.compute_metrics(
                        y_test, y_pred_test, y_proba_test, label_names
                    )
                    # Log test metrics to MLflow with "test_" prefix
                    for k, v in test_metrics.items():
                        if isinstance(v, (int, float)):
                            mlflow_module.log_metric(f"test_{k}", float(v))

                    logger.info(
                        "[%s] Test accuracy: %.4f  F1-macro: %.4f",
                        run.name,
                        test_metrics.get("val_accuracy", float("nan")),
                        test_metrics.get("val_f1_macro", float("nan")),
                    )
            except Exception as exc:
                logger.warning("[%s] Test-set evaluation failed: %s", run.name, exc)

        # ── 7. Summary ────────────────────────────────────────────────────
        acc = result.metrics.get("val_accuracy", float("nan"))
        f1  = result.metrics.get("val_f1_macro", float("nan"))
        logger.info(
            "[%s] Done — val_accuracy=%.4f  val_f1_macro=%.4f  size=%.1f KB",
            run.name, acc, f1, result.model_size_kb,
        )
        logger.info("[%s] Artefacts: %s", run.name, result.output_dir)


# ---------------------------------------------------------------------------
# Auto-selection helper (Stage 5 pre-opt, called at end of every sweep)
# ---------------------------------------------------------------------------

def _auto_select(
    experiment:   str,
    mlflow_uri:   Optional[str],
    output_dir:   Path,
    metric:       str            = "val_f1_macro",
    min_accuracy: Optional[float] = None,
    top_n:        int            = 5,
    n_runs:       int            = 1,
) -> None:
    """Run pre-opt model selection and write ``shortlist.json`` to *output_dir*.

    Skipped when *n_runs* <= 1 (single-model runs don't need a shortlist).
    Failures are non-fatal — a warning is logged and training is considered
    successful regardless.
    """
    if n_runs <= 1:
        logger.debug("Auto-select skipped (single-model run).")
        return
    from src.training.select import select_preopt, write_shortlist
    try:
        candidates = select_preopt(
            experiment   = experiment,
            mlflow_uri   = mlflow_uri,
            metric       = metric,
            min_accuracy = min_accuracy,
            top_n        = top_n,
        )
        if candidates:
            # Stable, experiment-scoped copy — use this path for downstream stages
            safe_name = experiment.replace("/", "_").replace(" ", "_")
            scoped_path = output_dir / f"shortlist_{safe_name}.json"
            write_shortlist(candidates, scoped_path, experiment, metric)
            # Generic alias — overwritten by any later sweep; use scoped name for stability
            shortlist_path = output_dir / "shortlist.json"
            write_shortlist(candidates, shortlist_path, experiment, metric)
            logger.info("Shortlist → %s", scoped_path)
            logger.warning(
                "shortlist.json is a convenience alias; it will be overwritten by future sweeps. "
                "Use shortlist_%s.json for stable references.", safe_name,
            )
        else:
            logger.warning("Auto-select: no qualifying runs found in experiment '%s'.", experiment)
    except Exception as exc:
        logger.warning("Auto-select failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_param(s: str) -> tuple[str, object]:
    """Parse ``key=value`` into a typed (key, value) pair."""
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"--param must be 'key=value', got '{s}'")
    k, v = s.split("=", 1)
    # Try int → float → str
    for cast in (int, float):
        try:
            return k.strip(), cast(v.strip())
        except ValueError:
            pass
    # Bool special-case
    if v.strip().lower() in ("true", "yes"):
        return k.strip(), True
    if v.strip().lower() in ("false", "no"):
        return k.strip(), False
    return k.strip(), v.strip()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.training.train",
        description="Stage 3 — Model Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Config-driven mode
    p.add_argument(
        "--config", metavar="YAML",
        help="Path to a training.yaml config (multi-model sweep).",
    )

    # Flag-based single-run mode
    single = p.add_argument_group("single-run flags (ignored when --config is used)")
    single.add_argument(
        "--features", metavar="DIR",
        help="Path to a Stage 2 FeatureSet directory.",
    )
    single.add_argument(
        "--features-test", metavar="DIR",
        help="Optional path to a held-out test FeatureSet.",
    )
    single.add_argument(
        "--model", metavar="NAME",
        help=f"Trainer name. One of: {', '.join(list_models())}",
    )
    single.add_argument(
        "--output", metavar="DIR", default="data/models",
        help="Root output directory for model artefacts (default: data/models).",
    )
    single.add_argument(
        "--val-split", type=float, default=0.2,
        help="Fraction of data used for validation (default: 0.2).",
    )
    single.add_argument(
        "--experiment", default="ml-pipeline",
        help="MLflow experiment name (default: ml-pipeline).",
    )
    single.add_argument(
        "--run-name", metavar="NAME",
        help="Human-readable run name (default: <model>).",
    )
    single.add_argument(
        "--max-samples", type=int, metavar="N",
        help="Randomly subsample to at most N examples (useful for smoke tests).",
    )
    single.add_argument(
        "--param", action="append", dest="params", metavar="KEY=VALUE",
        type=_parse_param, default=[],
        help="Trainer hyperparameter (repeat for multiple). E.g. --param C=10.0",
    )

    p.add_argument(
        "--no-auto-select", action="store_true",
        help="Skip automatic shortlist generation (Stage 5 pre-opt) after training.",
    )

    return p


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # ── Config-driven mode ────────────────────────────────────────────────
    if args.config:
        cfg = load_train_config(Path(args.config))
        runs = cfg.resolved_runs()
        if not runs:
            logger.error("No runs defined in %s", args.config)
            sys.exit(1)
        logger.info("Config sweep: %d run(s) in experiment '%s'", len(runs), cfg.experiment)
        for run in runs:
            try:
                _run_one(run, cfg.experiment, cfg.mlflow_uri)
            except Exception as exc:
                logger.error("Run '%s' failed: %s", run.name, exc, exc_info=True)

        if cfg.auto_select and not args.no_auto_select:
            _auto_select(
                experiment   = cfg.experiment,
                mlflow_uri   = cfg.mlflow_uri,
                output_dir   = Path(cfg.output_dir),
                metric       = cfg.auto_select_metric,
                min_accuracy = cfg.auto_select_min_accuracy,
                top_n        = cfg.auto_select_top_n,
                n_runs       = len(runs),
            )
        return

    # ── Flag-based single-run mode ────────────────────────────────────────
    if not args.features:
        parser.error("--features is required when not using --config")
    if not args.model:
        parser.error(f"--model is required. Available: {', '.join(list_models())}")

    params = dict(args.params) if args.params else {}
    run = ModelRunConfig(
        model             = args.model,
        name              = args.run_name or args.model,
        features_dir      = args.features,
        features_test_dir = args.features_test,
        output_dir        = args.output,
        val_split         = args.val_split,
        params            = params,
    )

    _run_one(run, args.experiment, mlflow_uri=None, max_samples=args.max_samples)


if __name__ == "__main__":
    main()