"""
Stage 7a — Convert best Keras model to TFLite for OpenMV deployment.

Benchmarks four quantization modes (fp32, dynamic INT8, static INT8, float16),
evaluates each on a held-out validation set, selects the best (smallest within
an accuracy-drop threshold), and writes a self-contained deployment bundle.

Output layout (``output_dir/``)
--------------------------------
    model_fp32.tflite
    model_dynamic.tflite
    model_int8.tflite
    model_float16.tflite
    deploy/
        model.tflite        best mode — copy this to Nicla Vision flash
        label_names.json    class names in label-index order
        mel_params.json     feature extraction parameters for nicla_main.py
    compilation_report.json

Usage
-----
    python -m src.compilation.to_tflite \\
        --model        data/models/tuned/.../trial_75/model.keras \\
        --features     data/processed/fsc22_melspec_augmented_train \\
        --features-val data/processed/fsc22_melspec_val \\
        --output       data/models/compiled/fsc22_12_classes \\
        --class-filter Axe BirdChirping Chainsaw Clapping Fire Firework \\
                       Footsteps Frog Generator Gunshot Handsaw Helicopter

    python -m src.compilation.to_tflite --config config/compilation.yaml

Config schema (compilation.yaml)
---------------------------------
::

    model:        data/models/tuned/.../trial_75/model.keras
    features:     data/processed/fsc22_melspec_augmented_train
    features_val: data/processed/fsc22_melspec_val
    output:       data/models/compiled/fsc22_12_classes
    class_filter: [Axe, BirdChirping, ...]   # optional — inferred from model if omitted
    max_accuracy_drop: 0.02                  # INT8 tolerance (default 0.02)

    # Mel-spectrogram parameters written to deploy/mel_params.json.
    # Defaults match AudioMelSpectrogram extractor defaults.
    sample_rate: 16000
    n_mels:      40
    n_fft:       512
    hop_length:  160
    duration:    5.0
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.optimization.quantize import (
    convert_to_tflite_dynamic,
    convert_to_tflite_float16,
    convert_to_tflite_fp32,
    convert_to_tflite_int8,
    evaluate_tflite,
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
# Helpers
# ---------------------------------------------------------------------------

def _apply_class_filter(
    X: np.ndarray,
    y: np.ndarray,
    label_names: list,
    class_filter: Optional[list],
) -> tuple[np.ndarray, np.ndarray, list]:
    """Restrict X/y/label_names to class_filter, remapping labels to 0..N-1.

    Classes are sorted alphabetically before assigning integer indices so the
    0..N-1 encoding is identical regardless of which loader produced the
    feature set (e.g. audio_folder yields alphabetical order; FSC22Loader uses
    metadata order).  This matches the convention in tune.py.
    """
    if not class_filter:
        return X, y, label_names
    filter_set = set(class_filter)
    # Sort by class name for a canonical, loader-independent encoding.
    allowed_pairs = sorted(
        [(i, n) for i, n in enumerate(label_names) if n in filter_set],
        key=lambda p: p[1],
    )
    allowed_idx = [i for i, _ in allowed_pairs]
    if not allowed_idx:
        raise ValueError(f"class_filter matched no classes in {label_names}")
    missing = filter_set - {label_names[i] for i in allowed_idx}
    if missing:
        logger.warning("class_filter: classes not found in dataset: %s", sorted(missing))
    mask    = np.isin(y, allowed_idx)
    X, y    = X[mask], y[mask]
    idx_map = {old: new for new, old in enumerate(allowed_idx)}
    y       = np.array([idx_map[lbl] for lbl in y], dtype=y.dtype)
    label_names = [i[1] for i in allowed_pairs]
    logger.info("class_filter applied — %d classes, %d samples", len(label_names), len(X))
    return X, y, label_names


def _generate_mel_filterbank(mel_params: dict, output_path: Path) -> None:
    """Save the librosa mel filterbank matrix to *output_path* as float32 .npy.

    Shape: ``(n_mels, n_fft // 2 + 1)``  e.g. ``(40, 257)``.

    This matrix is loaded on the Nicla at boot and used for on-device
    spectrogram computation, guaranteeing identical filterbank to training.
    """
    import librosa

    fb = librosa.filters.mel(
        sr=mel_params["sample_rate"],
        n_fft=mel_params["n_fft"],
        n_mels=mel_params["n_mels"],
    ).astype(np.float32)   # (n_mels, n_fft//2 + 1)
    np.save(str(output_path), fb)
    logger.info(
        "Mel filterbank %s → %s  (%.1f KB)",
        fb.shape, output_path.name, output_path.stat().st_size / 1024,
    )


def _infer_n_classes(model_path: Path) -> int:
    """Load the Keras model and return its output size (number of classes)."""
    import tensorflow as tf
    model = tf.keras.models.load_model(str(model_path))
    return int(model.output_shape[-1])


# ---------------------------------------------------------------------------
# Core conversion + evaluation loop
# ---------------------------------------------------------------------------

def compile_model(
    model_path:        Path,
    features_dir:      Path,
    features_val_dir:  Path,
    output_dir:        Path,
    class_filter:      Optional[list],
    mel_params:        dict,
    max_accuracy_drop: float = 0.02,
    forced_mode:       Optional[str] = None,
) -> dict:
    """Run all four TFLite modes; evaluate; select best; write deployment bundle.

    Returns the compilation report dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load calibration features (from training set) ────────────────────────
    logger.info("Loading calibration features from %s", features_dir)
    calib_fs    = FeaturePipeline.load(features_dir)
    X_calib     = calib_fs.features
    label_names = calib_fs.label_names or []

    # ── Load validation features ──────────────────────────────────────────────
    logger.info("Loading validation features from %s", features_val_dir)
    val_fs  = FeaturePipeline.load(features_val_dir)
    X_val   = val_fs.features
    y_val   = val_fs.labels
    val_lns = val_fs.label_names or []

    if y_val is None:
        raise ValueError("Validation FeatureSet has no labels.")

    # Infer class_filter from model output size if not provided
    if not class_filter:
        n_model_classes = _infer_n_classes(model_path)
        if n_model_classes < len(label_names):
            raise ValueError(
                f"Model has {n_model_classes} output classes but no --class-filter "
                f"was provided and the feature set has {len(label_names)} classes. "
                f"Pass --class-filter to specify which {n_model_classes} classes to evaluate."
            )

    # Calibration subset — filtered to the same classes.
    if calib_fs.labels is not None:
        X_calib_f, _, filtered_labels = _apply_class_filter(
            X_calib, calib_fs.labels, label_names, class_filter
        )
    else:
        X_calib_f = X_calib
        filtered_labels = sorted(class_filter) if class_filter else label_names

    # Apply class filter to val set.  _apply_class_filter sorts classes
    # alphabetically, so the 0..N-1 encoding matches training regardless of
    # which loader produced each feature set.
    X_val_f, y_val_f, filtered_labels = _apply_class_filter(
        X_val, y_val, val_lns, class_filter
    )
    logger.info(
        "Validation: %d samples  %d classes  shape %s",
        len(X_val_f), len(filtered_labels), X_val_f.shape[1:],
    )

    # ── Convert all modes ─────────────────────────────────────────────────────
    paths = {
        "fp32":    output_dir / "model_fp32.tflite",
        "dynamic": output_dir / "model_dynamic.tflite",
        "int8":    output_dir / "model_int8.tflite",
        "float16": output_dir / "model_float16.tflite",
    }

    logger.info("Converting to TFLite fp32 …")
    convert_to_tflite_fp32(model_path, paths["fp32"])

    logger.info("Converting to TFLite dynamic INT8 …")
    convert_to_tflite_dynamic(model_path, paths["dynamic"])

    logger.info("Converting to TFLite static INT8 (calibrating) …")
    convert_to_tflite_int8(model_path, paths["int8"], X_calib_f)

    logger.info("Converting to TFLite float16 …")
    convert_to_tflite_float16(model_path, paths["float16"])

    # ── Evaluate all modes ────────────────────────────────────────────────────
    logger.info("Evaluating all modes on validation set …")
    results = {}
    fp32_accuracy = None

    for mode, tflite_path in paths.items():
        metrics  = evaluate_tflite(tflite_path, X_val_f, y_val_f)
        size_kb  = tflite_path.stat().st_size / 1024
        acc      = metrics["accuracy"]
        lat      = metrics["latency_ms"]
        if mode == "fp32":
            fp32_accuracy = acc
        results[mode] = {
            "accuracy":   acc,
            "latency_ms": lat,
            "size_kb":    size_kb,
        }
        logger.info(
            "  %-10s  accuracy=%.4f  latency=%.2f ms  size=%.1f KB",
            mode, acc, lat, size_kb,
        )

    # ── Select best mode ──────────────────────────────────────────────────────
    if forced_mode:
        if forced_mode not in results:
            raise ValueError(f"--mode {forced_mode!r} is not a valid mode. Choose from: {list(results)}")
        best_mode = forced_mode
        logger.info("Mode forced to: %s", best_mode)
    else:
        # Prefer smallest model within accuracy_drop <= max_accuracy_drop vs fp32.
        # fp32 is always the fallback.
        best_mode = "fp32"
        for mode in ("int8", "dynamic", "float16"):
            drop = fp32_accuracy - results[mode]["accuracy"]
            if drop <= max_accuracy_drop:
                if results[mode]["size_kb"] < results[best_mode]["size_kb"]:
                    best_mode = mode

    best = results[best_mode]
    logger.info(
        "Best mode: %s  (accuracy=%.4f  drop=%.4f  size=%.1f KB)",
        best_mode,
        best["accuracy"],
        fp32_accuracy - best["accuracy"],
        best["size_kb"],
    )

    # ── Write deployment bundle ───────────────────────────────────────────────
    deploy_dir = output_dir / "deploy"
    deploy_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(paths[best_mode], deploy_dir / "model.tflite")
    logger.info("Deployment model → %s", deploy_dir / "model.tflite")

    (deploy_dir / "label_names.json").write_text(
        json.dumps(filtered_labels, indent=2)
    )

    (deploy_dir / "mel_params.json").write_text(
        json.dumps(mel_params, indent=2)
    )

    _generate_mel_filterbank(mel_params, deploy_dir / "mel_filterbank.npy")

    logger.info("Deployment bundle → %s", deploy_dir)

    # ── Write compilation report ──────────────────────────────────────────────
    report = {
        "timestamp":         datetime.now().isoformat(timespec="seconds"),
        "model_path":        str(model_path),
        "features_dir":      str(features_dir),
        "features_val_dir":  str(features_val_dir),
        "class_filter":      class_filter or None,
        "label_names":       filtered_labels,
        "n_classes":         len(filtered_labels),
        "n_val_samples":     int(len(y_val_f)),
        "max_accuracy_drop": max_accuracy_drop,
        "fp32_accuracy":     fp32_accuracy,
        "best_mode":         best_mode,
        "best_accuracy":     best["accuracy"],
        "best_accuracy_drop": float(fp32_accuracy - best["accuracy"]),
        "best_size_kb":      best["size_kb"],
        "best_latency_ms":   best["latency_ms"],
        "mel_params":        mel_params,
        "modes":             results,
        "deploy_dir":        str(deploy_dir),
    }
    report_path = output_dir / "compilation_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("Compilation report → %s", report_path)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.compilation.to_tflite",
        description="Stage 7a — Convert Keras model to TFLite for OpenMV deployment.",
    )
    parser.add_argument("--config",        metavar="YAML",  help="YAML config file.")
    parser.add_argument("--model",         metavar="PATH",  help="Path to model.keras.")
    parser.add_argument("--features",      metavar="DIR",   help="Calibration features dir.")
    parser.add_argument("--features-val",  metavar="DIR",   help="Validation features dir.")
    parser.add_argument("--output",        metavar="DIR",   help="Output directory.")
    parser.add_argument(
        "--class-filter", metavar="CLASS", nargs="+",
        help="Class names to include (must match training-time filter).",
    )
    parser.add_argument(
        "--max-accuracy-drop", type=float, default=0.02,
        help="Max tolerated accuracy drop vs fp32 for quantized selection (default 0.02).",
    )
    parser.add_argument(
        "--mode", metavar="MODE", choices=["fp32", "dynamic", "int8", "float16"],
        help="Force a specific TFLite mode instead of auto-selecting. "
             "Choices: fp32, dynamic, int8, float16.",
    )
    # Mel-spectrogram parameters
    parser.add_argument("--sample-rate", type=int,   default=16000)
    parser.add_argument("--n-mels",      type=int,   default=40)
    parser.add_argument("--n-fft",       type=int,   default=512)
    parser.add_argument("--hop-length",  type=int,   default=160)
    parser.add_argument("--duration",    type=float, default=5.0)
    args = parser.parse_args(argv)

    # Merge config file + CLI args (CLI takes precedence)
    cfg: dict = {}
    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text()) or {}

    def _get(key, cli_val, default=None):
        return cli_val if cli_val is not None else cfg.get(key, default)

    model_path       = Path(_get("model",        args.model))
    features_dir     = Path(_get("features",     args.features))
    features_val_dir = Path(_get("features_val", args.features_val))
    output_dir       = Path(_get("output",       args.output))
    class_filter     = args.class_filter or cfg.get("class_filter") or None
    max_drop         = float(_get("max_accuracy_drop", args.max_accuracy_drop, 0.02))
    forced_mode      = args.mode or cfg.get("mode") or None

    mel_params = {
        "sample_rate": int(_get("sample_rate", args.sample_rate, 16000)),
        "n_mels":      int(_get("n_mels",      args.n_mels,      40)),
        "n_fft":       int(_get("n_fft",       args.n_fft,       512)),
        "hop_length":  int(_get("hop_length",  args.hop_length,  160)),
        "duration":    float(_get("duration",  args.duration,    5.0)),
    }
    mel_params["input_shape"] = [
        mel_params["n_mels"],
        int(mel_params["duration"] * mel_params["sample_rate"] / mel_params["hop_length"]) + 1,
    ]

    for label, p in [("model", model_path), ("features", features_dir),
                     ("features_val", features_val_dir)]:
        if not p.exists():
            logger.error("%s not found: %s", label, p)
            sys.exit(1)

    logger.info("Model:          %s", model_path)
    logger.info("Features:       %s", features_dir)
    logger.info("Features val:   %s", features_val_dir)
    logger.info("Output:         %s", output_dir)
    logger.info("Class filter:   %s", class_filter or "(none)")
    logger.info("Mel params:     %s", mel_params)
    logger.info("Max drop:       %.3f", max_drop)

    report = compile_model(
        model_path        = model_path,
        features_dir      = features_dir,
        features_val_dir  = features_val_dir,
        output_dir        = output_dir,
        class_filter      = class_filter,
        mel_params        = mel_params,
        max_accuracy_drop = max_drop,
        forced_mode       = forced_mode,
    )

    logger.info("═" * 56)
    logger.info("Best mode:    %s", report["best_mode"])
    logger.info("Accuracy:     %.4f  (drop %.4f vs fp32)", report["best_accuracy"], report["best_accuracy_drop"])
    logger.info("Size:         %.1f KB", report["best_size_kb"])
    logger.info("Latency:      %.2f ms/sample", report["best_latency_ms"])
    logger.info("Deploy dir:   %s", report["deploy_dir"])
    logger.info("═" * 56)


if __name__ == "__main__":
    main()
