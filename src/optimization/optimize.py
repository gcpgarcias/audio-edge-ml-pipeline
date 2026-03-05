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
        --features        data/processed/birdeep_classical_train \\
        --output-dir      data/models/optimized \\
        --experiment      birdeep-optimization

Output layout (``output_dir/<model>/``)
----------------------------------------
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
    detect_model_type,
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
) -> Optional[dict]:
    """Run all four optimization modes for one shortlist candidate.

    Returns an optimization_report dict, or None if all modes failed.
    """
    model_name   = candidate["model"]
    run_id       = candidate.get("run_id", "")
    run_name     = candidate.get("run_name", model_name)
    artifact_uri = candidate.get("artifact_uri", "")
    val_acc_orig = candidate.get("val_accuracy", 0.0)

    model_dir = output_dir / model_name
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

    fp32_metrics = evaluate_onnx(fp32_path, X, y, label_names)
    logger.info(
        "[%s] fp32        acc=%.4f  latency=%.3f ms  size=%.1f KB",
        model_name, fp32_metrics["accuracy"], fp32_metrics["latency_ms"],
        fp32_path.stat().st_size / 1024,
    )

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
            fp32_path, model_dir / "model_static_int8.onnx", X
        ),
        "float16": lambda: optimize_float16(
            fp32_path, model_dir / "model_float16.onnx"
        ),
    }

    for mode_key, fn in _mode_fns.items():
        try:
            opt_path = fn()
            m = evaluate_onnx(opt_path, X, y, label_names)
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

    # ── 4. Select best mode ───────────────────────────────────────────────
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
        "[%s] Best mode: %s  (%.1f KB, acc=%.4f, drop=%.4f)",
        model_name, best_key, best["size_kb"],
        best["accuracy"], reference_acc - best["accuracy"],
    )

    # ── 5. Build report ───────────────────────────────────────────────────
    benchmark_results = {
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
        "original_size_kb":       original_size_kb,
        "val_accuracy_original":  val_acc_orig,
        "benchmark_results":      benchmark_results,
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

    # ── 6. MLflow logging ─────────────────────────────────────────────────
    if mlflow_module is not None:
        with mlflow_module.start_run(run_name=f"opt_{model_name}"):
            mlflow_module.log_params({
                "model":                       model_name,
                "original_run_id":             run_id,
                "selected_quantization_method": best_key,
                "max_accuracy_drop_threshold": max_accuracy_drop,
            })
            for mode_key, m in benchmark_results.items():
                mlflow_module.log_metric(f"{mode_key}_size_kb",    m["size_kb"])
                mlflow_module.log_metric(f"{mode_key}_accuracy",   m["accuracy"])
                mlflow_module.log_metric(f"{mode_key}_latency_ms", m["latency_ms"])
            mlflow_module.log_metric("optimized_size_kb",      best["size_kb"])
            mlflow_module.log_metric("compression_ratio",      original_size_kb / best["size_kb"])
            mlflow_module.log_metric("val_accuracy_optimized", best["accuracy"])
            mlflow_module.log_metric("accuracy_drop",          reference_acc - best["accuracy"])
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
        default="data/models/tuned/shortlist.json",
        help="Path to shortlist.json from Stage 6a (or 5a).  "
             "Default: data/models/tuned/shortlist.json",
    )
    parser.add_argument(
        "--features", metavar="DIR", required=True,
        help="Path to a FeatureSet directory used for evaluation and INT8 "
             "calibration.  Use the test split if available, otherwise the "
             "training split is acceptable.",
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

    shortlist_path = Path(args.shortlist)
    if not shortlist_path.exists():
        logger.error("Shortlist not found: %s", shortlist_path)
        sys.exit(1)

    features_dir = Path(args.features)
    if not features_dir.exists():
        logger.error("Features directory not found: %s", features_dir)
        sys.exit(1)

    output_dir = Path(args.output_dir)

    # ── Load features ──────────────────────────────────────────────────────
    logger.info("Loading features from %s", features_dir)
    feature_set  = FeaturePipeline.load(features_dir)
    X            = feature_set.features.reshape(len(feature_set.features), -1).astype(np.float32)
    y            = feature_set.labels
    label_names  = feature_set.label_names or []
    n_features   = X.shape[1]

    if y is None:
        logger.error("FeatureSet has no labels. Evaluation requires labelled data.")
        sys.exit(1)

    logger.info(
        "Features: %d samples  %d classes  n_features=%d",
        len(X), len(label_names), n_features,
    )

    # ── Load shortlist ─────────────────────────────────────────────────────
    shortlist_doc  = json.loads(shortlist_path.read_text())
    candidates     = shortlist_doc.get("candidates", [])
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
            )
            if report:
                reports.append(report)
        except Exception as exc:
            logger.error("Optimization of '%s' failed: %s", model_name, exc, exc_info=True)

    if not reports:
        logger.error("All optimization runs failed.")
        sys.exit(1)

    # ── Summary table ─────────────────────────────────────────────────────
    logger.info("─" * 60)
    logger.info(
        "  %-16s  %8s  %10s  %8s  %10s  %8s",
        "model", "orig_kb", "opt_kb", "ratio", "method", "acc_drop",
    )
    logger.info("─" * 60)
    for r in sorted(reports, key=lambda x: x["optimized_size_kb"]):
        logger.info(
            "  %-16s  %8.1f  %10.1f  %8.2f  %10s  %8.4f",
            r["model_name"],
            r["original_size_kb"],
            r["optimized_size_kb"],
            r["compression_ratio"],
            r["quantization_method"],
            r["accuracy_drop"],
        )
    logger.info("─" * 60)
    logger.info(
        "%d model(s) optimized. Run Stage 5c to select the best:\n"
        "  python -m src.training.select --post-opt \\\n"
        "    --shortlist %s \\\n"
        "    --opt-dir %s \\\n"
        "    --max-size-kb 256 --metric val_accuracy_optimized \\\n"
        "    --output data/models/best_model.json",
        len(reports), shortlist_path, output_dir,
    )


if __name__ == "__main__":
    main()
