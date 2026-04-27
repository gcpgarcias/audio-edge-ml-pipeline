"""
Stage 6b — ONNX-based resource optimization with multi-technique benchmarking.

Reads a shortlist.json, converts each model to ONNX, runs four optimization
modes (fp32 baseline, dynamic INT8, static INT8, float16), evaluates each on
a held-out feature set, selects the best mode (smallest model within an
accuracy-drop threshold), writes an optimization_report.json per model, and
logs all metrics to MLflow.

Usage
-----
    python -m src.optimization.optimize \\
        --shortlist       data/models/tuned/shortlist.json \\
        --output-dir      data/models/optimized \\
        --experiment      birdeep-optimization

Output layout (``output_dir/<experiment>/<model>/``)
-----------------------------------------------------
    model_fp32.onnx
    model_dynamic_int8.onnx
    model_static_int8.onnx
    model_float16.onnx
    optimization_report.json        consumed by Stage 5c select_postopt()

Stage 5c contract
-----------------
Each optimization_report.json must contain at minimum:
    run_id, run_name, model_name, original_model_path,
    optimized_model_path, original_size_kb, optimized_size_kb,
    compression_ratio, quantization_method, target_device,
    val_accuracy_original, val_accuracy_optimized, accuracy_drop,
    latency_ms, timestamp
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.optimization.quantize import (
    convert_to_onnx,
    evaluate_onnx,
    find_model_file,
    optimize_dynamic_int8,
    optimize_float16,
    optimize_static_int8,
)
from src.preprocessing.pipeline import FeaturePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)


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
# Per-candidate optimization
# ---------------------------------------------------------------------------

def _optimize_one(
    candidate:        dict,
    X:                np.ndarray,
    y:                np.ndarray,
    label_names:      list[str],
    n_features:       int,
    output_dir:       Path,
    max_accuracy_drop: float,
    mlflow_module,
    X_eval:           Optional[np.ndarray] = None,
    y_eval:           Optional[np.ndarray] = None,
) -> Optional[dict]:
    """Run ONNX and (for Keras) TFLite optimization modes for one shortlist candidate.

    Parameters
    ----------
    X, y:
        Features and labels used for INT8 calibration (typically training split).
    X_eval, y_eval:
        Features and labels used for accuracy measurement.  When ``None``,
        falls back to ``X`` / ``y`` (same data as calibration).  Pass a
        held-out validation set here to get accuracy numbers that are
        comparable to the tuning stage.

    Returns an optimization_report dict, or None if ONNX conversion failed.
    """
    if X_eval is None:
        X_eval = X
    if y_eval is None:
        y_eval = y
    model_name   = candidate["model"]
    run_id       = candidate.get("run_id", "")
    run_name     = candidate.get("run_name", model_name)
    artifact_uri = candidate.get("artifact_uri", "")
    val_acc_orig = candidate.get("val_accuracy", 0.0)

    # Use run_name as the output directory so that two candidates with the
    # same model type (e.g. cnn_mfcc vs cnn_melspec) don't overwrite each other.
    dir_key   = run_name if run_name and run_name != model_name else model_name
    model_dir = output_dir / dir_key
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Locate original model ──────────────────────────────────────────
    try:
        model_path = find_model_file(artifact_uri, model_name)
    except FileNotFoundError as exc:
        logger.error("[%s] Cannot find model file: %s", model_name, exc)
        return None

    original_size_kb = model_path.stat().st_size / 1024
    logger.info("[%s] Original model: %s  (%.1f KB)", model_name, model_path.name, original_size_kb)

    # ── 2. ONNX fp32 baseline ─────────────────────────────────────────────
    fp32_path = model_dir / "model_fp32.onnx"
    try:
        convert_to_onnx(model_path, n_features, fp32_path)
    except Exception as exc:
        logger.error("[%s] ONNX conversion failed: %s", model_name, exc)
        return None

    fp32_metrics = evaluate_onnx(fp32_path, X_eval, y_eval, label_names)
    logger.info(
        "[%s] fp32        acc=%.4f  latency=%.3f ms  size=%.1f KB",
        model_name, fp32_metrics["accuracy"], fp32_metrics["latency_ms"],
        fp32_path.stat().st_size / 1024,
    )
    # Re-evaluate the original model on X_eval via the fp32 ONNX (same weights,
    # negligible FP32 arithmetic difference).  This replaces the training-time
    # val_accuracy stored in the shortlist, which was measured on a different
    # (smaller) split and cannot be fairly compared to the optimized metrics.
    val_acc_orig_train = val_acc_orig   # keep the training-time value for reference
    val_acc_orig       = fp32_metrics["accuracy"]

    # ── 3. Optimization modes ─────────────────────────────────────────────
    modes: dict[str, dict] = {
        "fp32": {
            "path":       fp32_path,
            "size_kb":    fp32_path.stat().st_size / 1024,
            "accuracy":   fp32_metrics["accuracy"],
            "latency_ms": fp32_metrics["latency_ms"],
        }
    }

    _mode_fns = {
        "dynamic_int8": lambda: optimize_dynamic_int8(
            fp32_path, model_dir / "model_dynamic_int8.onnx"
        ),
        "static_int8": lambda: optimize_static_int8(
            fp32_path, model_dir / "model_static_int8.onnx", X  # X = training/calibration data
        ),
        "float16": lambda: optimize_float16(
            fp32_path, model_dir / "model_float16.onnx"
        ),
    }

    for mode_key, fn in _mode_fns.items():
        try:
            opt_path = fn()
            m = evaluate_onnx(opt_path, X_eval, y_eval, label_names)
            modes[mode_key] = {
                "path":       opt_path,
                "size_kb":    opt_path.stat().st_size / 1024,
                "accuracy":   m["accuracy"],
                "latency_ms": m["latency_ms"],
            }
            logger.info(
                "[%s] %-13s acc=%.4f  latency=%.3f ms  size=%.1f KB",
                model_name, mode_key,
                m["accuracy"], m["latency_ms"],
                modes[mode_key]["size_kb"],
            )
        except Exception as exc:
            logger.warning("[%s] Mode '%s' failed (skipping): %s", model_name, mode_key, exc)

    # ── 4. ONNX: select best mode ─────────────────────────────────────────
    # Smallest model where accuracy_drop ≤ max_accuracy_drop.
    # fp32 is always the fallback (accuracy_drop = 0).
    reference_acc = fp32_metrics["accuracy"]
    eligible = {
        k: v for k, v in modes.items()
        if reference_acc - v["accuracy"] <= max_accuracy_drop
    }
    if not eligible:
        eligible = {"fp32": modes["fp32"]}

    best_key = min(eligible, key=lambda k: eligible[k]["size_kb"])
    best     = modes[best_key]
    logger.info(
        "[%s] ONNX best:   %s  (%.1f KB, acc=%.4f, drop=%.4f)",
        model_name, best_key, best["size_kb"],
        best["accuracy"], reference_acc - best["accuracy"],
    )

    # ── 5. Build report ───────────────────────────────────────────────────
    onnx_benchmark_results = {
        k: {
            "size_kb":    v["size_kb"],
            "accuracy":   v["accuracy"],
            "latency_ms": v["latency_ms"],
            "path":       v["path"].name,
        }
        for k, v in modes.items()
    }
    report = {
        "run_id":                 run_id,
        "run_name":               run_name,
        "model_name":             model_name,
        "original_model_path":    str(model_path),
        "class_filter":           candidate.get("class_filter"),
        "feature_params":         candidate.get("feature_params"),
        "original_size_kb":            original_size_kb,
        "val_accuracy_original_train": val_acc_orig_train,
        "val_accuracy_original":       val_acc_orig,
        "benchmark_results":      onnx_benchmark_results,
        "optimized_model_path":   str(best["path"]),
        "optimized_size_kb":      best["size_kb"],
        "compression_ratio":      round(original_size_kb / best["size_kb"], 3),
        "quantization_method":    best_key,
        "target_device":          "arduino_nicla_vision",
        "val_accuracy_optimized": best["accuracy"],
        "accuracy_drop":          round(reference_acc - best["accuracy"], 6),
        "latency_ms":             best["latency_ms"],
        "timestamp":              datetime.now().isoformat(timespec="seconds"),
    }

    report_path = model_dir / "optimization_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("[%s] Report → %s", model_name, report_path)

    # ── 7. MLflow logging ─────────────────────────────────────────────────
    # One run per input model; every optimization mode is logged with a
    # mode-prefixed metric so all variants are visible in the MLflow UI.
    if mlflow_module is not None:
        with mlflow_module.start_run(run_name=f"opt_{dir_key}"):
            mlflow_module.log_params({
                "model":                       model_name,
                "original_run_id":             run_id,
                "onnx_best_mode":              best_key,
                "max_accuracy_drop_threshold": max_accuracy_drop,
            })
            # Original model
            mlflow_module.log_metric("original_size_kb",            original_size_kb)
            mlflow_module.log_metric("val_accuracy_original",       val_acc_orig)        # on eval set
            mlflow_module.log_metric("val_accuracy_original_train", val_acc_orig_train)  # training-time

            # All ONNX modes
            for mode_key, mv in modes.items():
                prefix = f"onnx_{mode_key}"
                mlflow_module.log_metric(f"{prefix}_size_kb",          mv["size_kb"])
                mlflow_module.log_metric(f"{prefix}_val_accuracy",     mv["accuracy"])
                mlflow_module.log_metric(f"{prefix}_latency_ms",       mv["latency_ms"])
                mlflow_module.log_metric(f"{prefix}_accuracy_drop",    reference_acc - mv["accuracy"])
                mlflow_module.log_metric(f"{prefix}_compression_ratio", original_size_kb / mv["size_kb"])

            # Summary: best ONNX selection
            mlflow_module.log_metric("onnx_best_size_kb",          best["size_kb"])
            mlflow_module.log_metric("onnx_best_val_accuracy",     best["accuracy"])
            mlflow_module.log_metric("onnx_best_latency_ms",       best["latency_ms"])
            mlflow_module.log_metric("onnx_best_accuracy_drop",    reference_acc - best["accuracy"])
            mlflow_module.log_metric("onnx_best_compression_ratio", original_size_kb / best["size_kb"])

            mlflow_module.log_artifact(str(best["path"]))
            mlflow_module.log_artifact(str(report_path))

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.optimization.optimize",
        description="Stage 6b — ONNX + onnxruntime multi-mode optimization benchmark",
    )
    parser.add_argument(
        "--shortlist", metavar="JSON",
        default=None,
        help="Path to shortlist.json from Stage 6a (or 5a).  "
             "Default: data/models/tuned/shortlist.json  "
             "Mutually exclusive with --model-path.",
    )
    parser.add_argument(
        "--model-path", metavar="PATH", default=None,
        dest="model_path",
        help="Path to a single .keras / .h5 / .joblib model file — skips the shortlist. "
             "Requires --features. Use --model-name, --class-filter, and --run-name to "
             "annotate the output report.",
    )
    parser.add_argument(
        "--model-name", metavar="NAME", default=None, dest="model_name",
        help="Trainer name for the --model-path candidate (e.g. 'cnn'). "
             "Defaults to the model file's parent directory name.",
    )
    parser.add_argument(
        "--run-name", metavar="NAME", default=None, dest="run_name",
        help="Run name for the --model-path candidate. "
             "Defaults to the model file's parent directory name.",
    )
    parser.add_argument(
        "--class-filter", nargs="+", default=None, metavar="CLASS",
        dest="class_filter",
        help="Class names to keep when using --model-path (e.g. Fire Silence Speaking).",
    )
    parser.add_argument(
        "--features", metavar="DIR", required=False, default=None,
        help="Path to a FeatureSet directory used for INT8 calibration.  "
             "Optional when every shortlist candidate already carries a "
             "'features_dir' field (logged by train.py).  Overrides "
             "per-candidate paths when supplied.",
    )
    parser.add_argument(
        "--features-eval", metavar="DIR", required=False, default=None,
        dest="features_eval",
        help="Path to a held-out FeatureSet directory used for accuracy "
             "measurement (e.g. a val or test split).  When omitted, "
             "accuracy is measured on the same features used for calibration, "
             "which inflates sklearn results and gives non-comparable numbers. "
             "Only candidates whose feature shape matches this directory will "
             "use it; others fall back to calibration features.",
    )
    parser.add_argument(
        "--output-dir", metavar="DIR",
        default="data/models/optimized",
        help="Root directory for optimized models and reports.  "
             "Default: data/models/optimized",
    )
    parser.add_argument(
        "--experiment", metavar="NAME",
        default="birdeep-optimization",
        help="MLflow experiment name.  Default: birdeep-optimization",
    )
    parser.add_argument(
        "--mlflow-uri", metavar="URI", default=None,
        help="MLflow tracking URI.  Default: env MLFLOW_TRACKING_URI or 'mlruns/'",
    )
    parser.add_argument(
        "--max-accuracy-drop", type=float, default=0.05, metavar="FLOAT",
        help="Maximum allowed accuracy drop vs fp32 baseline when choosing "
             "the best optimized mode.  Default: 0.05",
    )
    args = parser.parse_args(argv)

    if args.model_path and args.shortlist:
        parser.error("--model-path and --shortlist are mutually exclusive.")
    if not args.model_path and not args.shortlist:
        args.shortlist = "data/models/tuned/shortlist.json"

    safe_experiment = args.experiment.replace("/", "_").replace(" ", "_")
    output_dir = Path(args.output_dir) / safe_experiment

    # ── Build candidates list ──────────────────────────────────────────────
    if args.model_path:
        model_path = Path(args.model_path)
        if not model_path.exists():
            logger.error("Model not found: %s", model_path)
            sys.exit(1)
        run_name   = args.run_name   or model_path.parent.name
        model_name = args.model_name or model_path.parent.name
        candidates = [{
            "run_id":       "manual",
            "run_name":     run_name,
            "model":        model_name,
            "artifact_uri": str(model_path.parent),
            "features_dir": args.features,
            "class_filter": args.class_filter,
        }]
        logger.info("Single model: %s  (%s)", model_path, model_name)
    else:
        shortlist_path = Path(args.shortlist)
        if not shortlist_path.exists():
            logger.error("Shortlist not found: %s", shortlist_path)
            sys.exit(1)
        shortlist_doc = json.loads(shortlist_path.read_text())
        candidates    = shortlist_doc.get("candidates", [])
        if not candidates:
            logger.error("No candidates in shortlist: %s", shortlist_path)
            sys.exit(1)
        logger.info(
            "Shortlist: %d candidate(s) from experiment '%s'",
            len(candidates), shortlist_doc.get("experiment", "?"),
        )

    # ── MLflow ────────────────────────────────────────────────────────────
    mlflow_module = _setup_mlflow(args.mlflow_uri, args.experiment)

    # ── Optimize each candidate ───────────────────────────────────────────
    reports = []
    for candidate in candidates:
        model_name = candidate.get("model", "unknown")
        logger.info("─" * 60)
        logger.info("Optimizing: %s", model_name)

        # Resolve features directory: per-candidate field takes precedence;
        # --features acts as a global override / fallback.
        candidate_features = candidate.get("features_dir")
        resolved_features  = args.features or candidate_features
        if not resolved_features:
            logger.error(
                "[%s] No features directory available — pass --features or re-run "
                "training so 'features_dir' is stored in the shortlist.",
                model_name,
            )
            continue
        features_dir = Path(resolved_features)
        if not features_dir.exists():
            logger.error("[%s] Features directory not found: %s", model_name, features_dir)
            continue

        logger.info("[%s] Loading calibration features from %s", model_name, features_dir)
        feature_set = FeaturePipeline.load(features_dir)
        X           = feature_set.features.reshape(len(feature_set.features), -1).astype(np.float32)
        y           = feature_set.labels
        label_names = feature_set.label_names or []
        n_features  = X.shape[1]

        if y is None:
            logger.error("[%s] FeatureSet has no labels. Skipping.", model_name)
            continue

        # Apply class_filter if the model was trained on a subset of classes.
        # Without this, label indices from a 27-class feature set are evaluated
        # against a model with e.g. 6 output classes, giving ~chance accuracy.
        class_filter_raw = candidate.get("class_filter")
        if class_filter_raw:
            import json as _json
            if isinstance(class_filter_raw, list):
                filter_set = set(class_filter_raw)
            else:
                filter_set = set(_json.loads(class_filter_raw))
            allowed_indices = [i for i, n in enumerate(label_names) if n in filter_set]
            if allowed_indices:
                mask        = np.isin(y, allowed_indices)
                X, y        = X[mask], y[mask]
                idx_map     = {old: new for new, old in enumerate(allowed_indices)}
                y           = np.array([idx_map[lbl] for lbl in y], dtype=y.dtype)
                label_names = [label_names[i] for i in allowed_indices]
                n_features  = X.shape[1]
                logger.info(
                    "[%s] class_filter applied: %d classes, %d samples (calibration)",
                    model_name, len(label_names), len(X),
                )

        logger.info(
            "[%s] Calibration features: %d samples  %d classes  n_features=%d",
            model_name, len(X), len(label_names), n_features,
        )

        # ── Load eval features (held-out set for accuracy measurement) ─────
        # Resolution order:
        #   1. --features-eval CLI arg (global override)
        #   2. features_eval_dir from the shortlist candidate (per-candidate)
        #   3. None → fallback to calibration features inside _optimize_one
        X_eval: Optional[np.ndarray] = None
        y_eval: Optional[np.ndarray] = None
        resolved_features_eval = (args.features_eval
                                   or candidate.get("features_eval_dir")
                                   or candidate.get("features_val"))
        if resolved_features_eval:
            eval_dir = Path(resolved_features_eval)
            if not eval_dir.exists():
                logger.warning(
                    "[%s] features_eval_dir not found: %s — falling back to calibration features",
                    model_name, eval_dir,
                )
            else:
                eval_fs           = FeaturePipeline.load(eval_dir)
                X_eval_raw        = eval_fs.features.reshape(len(eval_fs.features), -1).astype(np.float32)
                y_eval_raw        = eval_fs.labels
                eval_label_names  = eval_fs.label_names or []
                if X_eval_raw.shape[1] != n_features:
                    logger.warning(
                        "[%s] --features-eval shape mismatch (%d vs %d) — "
                        "falling back to calibration features for accuracy",
                        model_name, X_eval_raw.shape[1], n_features,
                    )
                elif y_eval_raw is None:
                    logger.warning(
                        "[%s] --features-eval has no labels — falling back to calibration features",
                        model_name,
                    )
                else:
                    if class_filter_raw and label_names:
                        # Re-encode eval labels by class NAME using the training
                        # label ordering.  The eval and calibration sets may come
                        # from different loaders with different class orderings
                        # (e.g. audio_folder = alphabetical; FSC22Loader =
                        # dataset order).  Index-based remapping produces
                        # scrambled labels and near-chance accuracy.
                        train_name_to_idx = {n: i for i, n in enumerate(label_names)}
                        train_class_set   = set(label_names)
                        sample_names      = [eval_label_names[i] for i in y_eval_raw]
                        eval_mask         = np.array([n in train_class_set for n in sample_names])
                        X_eval            = X_eval_raw[eval_mask]
                        y_eval            = np.array(
                            [train_name_to_idx[n] for n in sample_names if n in train_class_set],
                            dtype=y_eval_raw.dtype,
                        )
                    else:
                        X_eval = X_eval_raw
                        y_eval = y_eval_raw
                    logger.info(
                        "[%s] Eval features: %d samples from %s",
                        model_name, len(X_eval), eval_dir,
                    )

        try:
            report = _optimize_one(
                candidate         = candidate,
                X                 = X,
                y                 = y,
                label_names       = label_names,
                n_features        = n_features,
                output_dir        = output_dir,
                max_accuracy_drop = args.max_accuracy_drop,
                mlflow_module     = mlflow_module,
                X_eval            = X_eval,
                y_eval            = y_eval,
            )
            if report:
                reports.append(report)
        except Exception as exc:
            logger.error("Optimization of '%s' failed: %s", model_name, exc, exc_info=True)

    if not reports:
        logger.error("All optimization runs failed.")
        sys.exit(1)

    # ── Summary table ─────────────────────────────────────────────────────
    sep = "─" * 90
    logger.info(sep)
    logger.info(
        "  %-40s  %8s  │  %10s  %14s  %8s  %10s",
        "run", "orig_kb", "onnx_kb", "method", "acc_drop", "lat_ms",
    )
    logger.info(sep)
    for r in sorted(reports, key=lambda x: x["optimized_size_kb"]):
        label = r.get("run_name") or r["model_name"]
        logger.info(
            "  %-40s  %8.1f  │  %10.1f  %14s  %8.4f  %10.3f",
            label,
            r["original_size_kb"],
            r["optimized_size_kb"],
            r["quantization_method"],
            r["accuracy_drop"],
            r["latency_ms"],
        )
    logger.info(sep)
    logger.info(
        "%d model(s) optimized → %s\n"
        "Run Stage 5c to select the best:\n"
        "  python -m src.training.select --post-opt\n"
        "    --shortlist %s\n"
        "    --opt-dir %s\n"
        "    --max-size-kb 256 --metric val_accuracy_optimized\n"
        "    --output data/models/best_model.json",
        len(reports), output_dir, args.shortlist or args.model_path, output_dir,
    )


if __name__ == "__main__":
    main()
