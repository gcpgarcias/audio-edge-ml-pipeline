"""
Stage 6a — Model fine-tuning via hyperparameter search.

Dispatches automatically by model type:
  • Classical (sklearn) → GridSearchCV with stratified k-fold CV
  • Deep (Keras)        → Random search over a discrete search_space

Reads a single YAML config with a ``runs:`` list. Each run specifies either
``grid:`` (classical) or ``search_space:`` (deep). The dispatcher resolves the
model type from the registered trainer class.

Usage
-----
    python -m src.training.tune --config config/tuning.yaml

Config schema
-------------
::

    output_dir:    data/models/tuned
    experiment:    birdeep-tuning
    mlflow_uri:    mlruns/
    val_split:     0.2

    # Classical defaults
    cv:      5
    scoring: f1_macro

    # Deep defaults
    n_trials:     10
    sweep_epochs: 25
    seed:         42

    # Optional: restrict to models listed in a prior shortlist
    # shortlist: data/models/shortlist_birdeep-classification.json

    runs:
      # Classical — grid: key triggers GridSearchCV
      - model: lda
        features_dir: data/processed/birdeep_classical_train
        grid:
          solver: [svd, lsqr]

      # Deep — search_space: key triggers random search
      - model:        mlp
        name:         birdeep_mlp_sweep
        features_dir: data/processed/birdeep_classical_train
        search_space:
          hidden_units:  [[64, 32], [128, 64], [256, 128]]
          dropout:       [0.2, 0.3, 0.4]
          batch_size:    [16, 32, 64]
          learning_rate: [0.001, 0.0005, 0.0001]

Output layout (``output_dir/<run_name>/``)
-------------------------------------------
Classical:  <model>.joblib  classification_report.txt  confusion_matrix.png  model_info.json
Deep:       trial_00/ … trial_NN/  trial_summary.json
Both:       shortlist.json + shortlist_<experiment>.json written to output_dir root
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.preprocessing.pipeline import FeaturePipeline
from src.training.models import get_model
from src.training.evaluate import (
    compute_metrics,
    log_run_to_mlflow,
    save_classification_report,
    save_confusion_matrix_png,
    save_model_info,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classical — GridSearchCV
# ---------------------------------------------------------------------------

def _build_estimator(model_name: str):
    """Return a fresh, unfitted sklearn estimator for *model_name*."""
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    factories = {
        "svm":           lambda: SVC(probability=True, class_weight="balanced"),
        "lda":           lambda: LinearDiscriminantAnalysis(),
        "decision_tree": lambda: DecisionTreeClassifier(class_weight="balanced"),
        "random_forest": lambda: RandomForestClassifier(
            class_weight="balanced", n_jobs=-1, random_state=42
        ),
        "knn":     lambda: KNeighborsClassifier(),
        "pca_svm": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("pca",    PCA(random_state=42)),
            ("svm",    SVC(probability=True, class_weight="balanced")),
        ]),
    }
    if model_name not in factories:
        raise ValueError(
            f"No estimator factory for '{model_name}'. Supported: {sorted(factories)}"
        )
    return factories[model_name]()


_PARAM_PREFIXES: dict[str, dict[str, str]] = {
    "pca_svm": {
        "n_components": "pca__n_components",
        "C":            "svm__C",
        "kernel":       "svm__kernel",
        "gamma":        "svm__gamma",
    },
}


def _remap_param_grid(model_name: str, param_grid: dict) -> dict:
    mapping = _PARAM_PREFIXES.get(model_name, {})
    return {mapping.get(k, k): v for k, v in param_grid.items()}


def _tune_classical(
    run_cfg:     dict,
    default_cfg: dict,
    mlflow_module,
) -> Optional[dict]:
    """GridSearchCV for one classical run. Returns a result dict or None."""
    from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

    model_name   = run_cfg["model"]
    run_label    = run_cfg.get("name") or model_name
    features_dir = Path(run_cfg.get("features_dir") or default_cfg.get("features_dir", ""))
    output_dir   = Path(run_cfg.get("output_dir")   or default_cfg["output_dir"]) / run_label
    val_split    = float(run_cfg.get("val_split")   or default_cfg.get("val_split", 0.2))
    cv           = int(run_cfg.get("cv")            or default_cfg.get("cv", 5))
    scoring      = str(run_cfg.get("scoring")       or default_cfg.get("scoring", "f1_macro"))
    param_grid   = run_cfg.get("grid") or {}

    logger.info("[%s] Loading features from %s", run_label, features_dir)
    feature_set = FeaturePipeline.load(features_dir)
    X, y        = feature_set.features, feature_set.labels
    label_names = feature_set.label_names or []

    if y is None:
        logger.error("[%s] FeatureSet has no labels — skipping.", run_label)
        return None

    X_flat = X.reshape(len(X), -1).astype(np.float32)
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X_flat, y, test_size=val_split, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_val, y_train, y_val = train_test_split(
            X_flat, y, test_size=val_split, random_state=42
        )

    logger.info(
        "[%s] Train: %d  Val: %d  Classes: %d",
        run_label, len(X_train), len(X_val), len(label_names),
    )

    n_combos = math.prod(len(v) for v in param_grid.values()) if param_grid else 1
    logger.info(
        "[%s] GridSearchCV: %d combination(s) × %d folds = %d fits",
        run_label, n_combos, cv, n_combos * cv,
    )

    estimator   = _build_estimator(model_name)
    remapped    = _remap_param_grid(model_name, param_grid)
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator, remapped,
        cv=cv_splitter,
        scoring=scoring,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    best_estimator = grid_search.best_estimator_
    best_params    = grid_search.best_params_
    cv_best_score  = float(grid_search.best_score_)

    logger.info(
        "[%s] Best CV %s = %.4f  →  %s",
        run_label, scoring, cv_best_score, dict(best_params),
    )

    y_pred_val  = best_estimator.predict(X_val)
    y_proba_val: Optional[np.ndarray] = None
    if hasattr(best_estimator, "predict_proba"):
        try:
            y_proba_val = best_estimator.predict_proba(X_val)
        except Exception:
            pass

    val_metrics = compute_metrics(y_val, y_pred_val, y_proba_val, label_names)
    logger.info(
        "[%s] Val  accuracy=%.4f  f1_macro=%.4f",
        run_label,
        val_metrics.get("val_accuracy",  float("nan")),
        val_metrics.get("val_f1_macro",  float("nan")),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path    = output_dir / f"{model_name}.joblib"
    joblib.dump(best_estimator, model_path)
    model_size_kb = model_path.stat().st_size / 1024

    run_name   = f"{run_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    params_str = {"model": model_name, **{k: str(v) for k, v in best_params.items()}}
    save_classification_report(y_val, y_pred_val, label_names,
                               output_dir / "classification_report.txt")
    save_confusion_matrix_png(val_metrics.get("confusion_matrix", []),
                              label_names, output_dir / "confusion_matrix.png")
    save_model_info(output_dir, model_name, run_name, val_metrics, params_str, model_size_kb)

    mlflow_params = {"model": model_name, "cv_folds": str(cv),
                     "cv_scoring": scoring, **params_str}
    log_metrics   = {**val_metrics, "cv_best_score": cv_best_score,
                     "model_size_kb": model_size_kb}

    with mlflow_module.start_run(run_name=run_name) as active_run:
        log_run_to_mlflow(active_run, mlflow_params, log_metrics, output_dir)
        import mlflow
        mlflow.log_artifact(str(model_path))

    return {
        "model":         model_name,
        "run_name":      run_name,
        "run_id":        active_run.info.run_id,
        "val_accuracy":  val_metrics.get("val_accuracy",  0.0),
        "val_f1_macro":  val_metrics.get("val_f1_macro",  0.0),
        "cv_best_score": cv_best_score,
        "model_size_kb": model_size_kb,
        "best_params":   params_str,
        "artifact_uri":  str(output_dir),
    }


# ---------------------------------------------------------------------------
# Deep — Random search
# ---------------------------------------------------------------------------

def _sample_params(search_space: dict, rng: random.Random) -> dict:
    return {k: rng.choice(v) for k, v in search_space.items()}


def _sweep_deep(
    run_cfg:     dict,
    default_cfg: dict,
    mlflow_module,
) -> Optional[dict]:
    """Random search for one deep model run. Returns the best trial dict or None."""
    from sklearn.model_selection import train_test_split

    model_name   = run_cfg["model"]
    run_label    = run_cfg.get("name") or model_name
    features_dir = Path(run_cfg.get("features_dir") or default_cfg.get("features_dir", ""))
    output_dir   = Path(run_cfg.get("output_dir")   or default_cfg["output_dir"]) / run_label
    val_split    = float(run_cfg.get("val_split")   or default_cfg.get("val_split", 0.2))
    n_trials     = int(run_cfg.get("n_trials")      or default_cfg.get("n_trials", 10))
    sweep_epochs = int(run_cfg.get("sweep_epochs")  or default_cfg.get("sweep_epochs", 25))
    seed         = int(default_cfg.get("seed", 42))
    search_space = run_cfg.get("search_space") or {}

    logger.info("[%s] Loading features from %s", run_label, features_dir)
    feature_set = FeaturePipeline.load(features_dir)
    X, y        = feature_set.features, feature_set.labels
    label_names = feature_set.label_names or []

    if y is None:
        logger.error("[%s] FeatureSet has no labels — skipping.", run_label)
        return None

    logger.info(
        "[%s] %d samples  %d classes  shape %s",
        run_label, len(X), len(label_names), X.shape[1:],
    )

    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_split, random_state=seed, stratify=y
        )
    except ValueError:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_split, random_state=seed
        )

    if search_space:
        space_size = 1
        for v in search_space.values():
            space_size *= len(v)
        logger.info(
            "[%s] Search space: %d combinations  running %d random trial(s)",
            run_label, space_size, n_trials,
        )
    else:
        logger.info("[%s] No search_space — 1 trial with defaults.", run_label)
        n_trials = 1

    rng           = random.Random(seed)
    best_score    = -1.0
    best_result   = None
    trial_results = []

    for trial_idx in range(n_trials):
        sampled        = _sample_params(search_space, rng) if search_space else {}
        trial_run_name = (
            f"{run_label}_trial{trial_idx:02d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        trial_dir = output_dir / f"trial_{trial_idx:02d}"

        logger.info("─" * 56)
        logger.info("[%s] Trial %d/%d  %s", run_label, trial_idx + 1, n_trials, sampled)

        try:
            trainer_cls = get_model(model_name)
            trainer     = trainer_cls(epochs=sweep_epochs, **sampled)

            with mlflow_module.start_run(run_name=trial_run_name) as active_run:
                result = trainer.fit(
                    X_train, y_train,
                    X_val, y_val,
                    label_names = label_names,
                    run_name    = trial_run_name,
                    output_dir  = trial_dir,
                    mlflow_run  = active_run,
                )

            score = result.metrics.get("val_f1_macro", 0.0)
            logger.info(
                "[%s] Trial %d  val_accuracy=%.4f  val_f1_macro=%.4f",
                run_label, trial_idx + 1,
                result.metrics.get("val_accuracy", float("nan")), score,
            )

            trial_record = {
                "trial":         trial_idx,
                "run_id":        result.run_id,
                "run_name":      trial_run_name,
                "model":         model_name,
                "val_accuracy":  result.metrics.get("val_accuracy",  0.0),
                "val_f1_macro":  score,
                "cv_best_score": None,
                "model_size_kb": result.model_size_kb,
                "best_params":   {k: str(v) for k, v in sampled.items()},
                "artifact_uri":  str(trial_dir),
            }
            trial_results.append(trial_record)

            if score > best_score:
                best_score  = score
                best_result = trial_record

        except Exception as exc:
            logger.error(
                "[%s] Trial %d failed: %s", run_label, trial_idx + 1, exc, exc_info=True
            )

    if best_result is None:
        logger.error("[%s] All trials failed.", run_label)
        return None

    logger.info(
        "[%s] Best trial %d  val_f1_macro=%.4f  params=%s",
        run_label, best_result["trial"] + 1, best_score, best_result["best_params"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "trial_summary.json").write_text(json.dumps({
        "run_name": run_label, "model": model_name,
        "n_trials": n_trials, "sweep_epochs": sweep_epochs,
        "best_trial": best_result["trial"], "best_val_f1_macro": best_score,
        "trials": trial_results,
    }, indent=2))

    return best_result


# ---------------------------------------------------------------------------
# MLflow helper
# ---------------------------------------------------------------------------

def _setup_mlflow(uri: Optional[str], experiment: str):
    import mlflow
    tracking_uri = uri or os.getenv("MLFLOW_TRACKING_URI", "mlruns/")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    logger.info("MLflow tracking URI: %s  experiment: %s", tracking_uri, experiment)
    return mlflow


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.training.tune",
        description="Stage 6a — Hyperparameter search (GridSearchCV for classical, "
                    "random search for deep)",
    )
    parser.add_argument(
        "--config", metavar="YAML", required=True,
        help="Path to tuning.yaml config file.",
    )
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logger.error("Config not found: %s", cfg_path)
        sys.exit(1)

    raw = yaml.safe_load(cfg_path.read_text()) or {}
    for key in ("output_dir", "runs"):
        if key not in raw:
            logger.error("tuning.yaml must include '%s'.", key)
            sys.exit(1)

    output_dir = Path(raw["output_dir"])
    experiment = raw.get("experiment", "birdeep-tuning")
    mlflow_uri = raw.get("mlflow_uri")
    runs: list = raw.get("runs") or []

    # Optional shortlist filter
    allowed_models: Optional[set[str]] = None
    shortlist_path = raw.get("shortlist")
    if shortlist_path:
        sl = json.loads(Path(shortlist_path).read_text())
        allowed_models = {c["model"] for c in sl.get("candidates", [])}
        logger.info("Shortlist filter active — tuning only: %s", sorted(allowed_models))

    eligible_runs = [
        r for r in runs
        if allowed_models is None or r.get("model") in allowed_models
    ]
    if not eligible_runs:
        logger.error("No eligible runs (check shortlist vs. run model names).")
        sys.exit(1)

    logger.info("Tuning %d run(s)", len(eligible_runs))
    mlflow_module = _setup_mlflow(mlflow_uri, experiment)

    results = []
    for run_cfg in eligible_runs:
        model_name = run_cfg.get("model", "?")
        run_label  = run_cfg.get("name") or model_name

        try:
            model_type = get_model(model_name).model_type
        except (KeyError, ValueError) as exc:
            logger.error("Unknown model '%s': %s", model_name, exc)
            continue

        logger.info("═" * 56)
        logger.info("Run: %-20s  type=%s", run_label, model_type)

        try:
            if model_type == "classical":
                if "grid" not in run_cfg:
                    logger.warning("[%s] No 'grid:' key — skipping.", run_label)
                    continue
                result = _tune_classical(run_cfg, raw, mlflow_module)
            else:
                if "search_space" not in run_cfg:
                    logger.warning("[%s] No 'search_space:' key — skipping.", run_label)
                    continue
                result = _sweep_deep(run_cfg, raw, mlflow_module)

            if result:
                results.append(result)

        except Exception as exc:
            logger.error("Run '%s' failed: %s", run_label, exc, exc_info=True)

    if not results:
        logger.error("All runs failed.")
        sys.exit(1)

    # Write shortlist (unified — both classical and deep)
    results.sort(key=lambda r: r.get("val_f1_macro", 0.0), reverse=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    shortlist_doc = {
        "experiment":   experiment,
        "metric":       "val_f1_macro",
        "n_candidates": len(results),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "candidates": [
            {
                "rank":          rank,
                "run_id":        r.get("run_id", ""),
                "run_name":      r.get("run_name", ""),
                "model":         r.get("model", ""),
                "val_accuracy":  r.get("val_accuracy",  0.0),
                "val_f1_macro":  r.get("val_f1_macro",  0.0),
                "cv_best_score": r.get("cv_best_score"),  # null for deep runs
                "model_size_kb": r.get("model_size_kb",  0.0),
                "best_params":   r.get("best_params", {}),
                "artifact_uri":  r.get("artifact_uri", ""),
            }
            for rank, r in enumerate(results, 1)
        ],
    }

    shortlist_out = output_dir / "shortlist.json"
    shortlist_out.write_text(json.dumps(shortlist_doc, indent=2))

    safe_name   = experiment.replace("/", "_").replace(" ", "_")
    scoped_path = output_dir / f"shortlists/shortlist_{safe_name}.json"
    scoped_path.write_text(json.dumps(shortlist_doc, indent=2))

    logger.info("Shortlist (%d candidates) → %s", len(results), shortlist_out)

    # Summary table
    logger.info("═" * 56)
    logger.info("  %-22s  %-12s  %8s  %8s", "run", "model", "val_acc", "f1_macro")
    logger.info("─" * 56)
    for r in results:
        logger.info(
            "  %-22s  %-12s  %8.4f  %8.4f",
            r.get("run_name", "")[:22], r.get("model", ""),
            r.get("val_accuracy", 0.0), r.get("val_f1_macro", 0.0),
        )
    logger.info("═" * 56)


if __name__ == "__main__":
    main()
