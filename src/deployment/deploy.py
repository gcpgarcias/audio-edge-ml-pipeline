"""
Stage 7 — C codegen + PlatformIO deployment.

Converts a trained Keras model to a self-contained PlatformIO C project
that runs on any Arduino-compatible microcontroller.

Usage
-----
    python -m src.deployment.deploy \\
        --model   data/models/tuned/fsc22_cnn/model.keras \\
        --board   nicla_vision \\
        --max-ram 256 \\
        --output  deploy/c_nicla

    python -m src.deployment.deploy \\
        --model   data/models/fsc22_cnn/model.keras \\
        --board   esp32s3 \\
        --max-ram 512 \\
        --output  deploy/c_esp32

Available boards
----------------
    nicla_vision   STM32H747  1 MB RAM  2 MB Flash   PDM mic
    nano_ble       nRF52840   256 KB    1 MB Flash   PDM mic
    esp32s3        ESP32-S3   512 KB    16 MB Flash  I2S mic
    pico2          RP2350     520 KB    4 MB Flash   I2S mic

--max-ram is checked against the peak activation arena.  If exceeded,
the command fails and reports the bottleneck layer with suggestions.

Feature extraction parameters are read from the processed feature set's
info.json (--features-dir) or from the last used extraction config.
Override individual params with --sample-rate, --n-fft, etc.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_model(model_path: Path):
    import tensorflow as tf
    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    model.summary()
    return model


def _load_labels(args) -> list[str]:
    # Explicit label file
    if args.labels:
        with open(args.labels) as f:
            labels = json.load(f)
        return _apply_class_filter(labels, args)

    # Auto-detect class_filter from optimization_report.json next to the ONNX
    if hasattr(args, "model") and Path(args.model).suffix.lower() == ".onnx":
        report_path = Path(args.model).parent / "optimization_report.json"
        if report_path.exists():
            with open(report_path) as f:
                report = json.load(f)
            cf = report.get("class_filter")
            if cf and not args.class_filter:
                args.class_filter = cf

    if args.features_dir:
        p = Path(args.features_dir) / "label_names.json"
        if p.exists():
            with open(p) as f:
                labels = json.load(f)
            return _apply_class_filter(labels, args)

    raise ValueError(
        "Could not determine class labels. "
        "Provide --labels <label_names.json> or --features-dir <processed_dir>."
    )


def _apply_class_filter(labels: list[str], args) -> list[str]:
    cf = getattr(args, "class_filter", None)
    if not cf:
        return labels
    allowed = set(cf)
    filtered = [l for l in labels if l in allowed]
    if not filtered:
        raise ValueError(f"class_filter {cf} matched none of {labels}")
    return filtered


def _load_feature_params(args) -> dict:
    """Build feature_params dict from CLI args + optional features-dir info."""
    base = {
        "sample_rate": 16000,
        "n_fft":       512,
        "hop_length":  160,
        "n_mels":      40,
        "n_mfcc":      40,
        "duration":    5.0,
        "n_features":  92,
    }

    # Pull stored feature_params from optimization report when using ONNX
    if hasattr(args, "model") and Path(args.model).suffix.lower() == ".onnx":
        report_path = Path(args.model).parent / "optimization_report.json"
        if report_path.exists():
            with open(report_path) as f:
                report = json.load(f)
            stored = report.get("feature_params") or {}
            for k in ("sample_rate", "n_fft", "hop_length", "n_mels", "n_mfcc", "duration"):
                if k in stored:
                    base[k] = stored[k]

    if args.features_dir:
        info_path = Path(args.features_dir) / "info.json"
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
            fs = info.get("feature_shape", [])
            if len(fs) >= 2:
                base["n_mels"]     = fs[0]
                base["n_features"] = fs[0]
                # Derive n_samples from n_frames using center=True inverse:
                # n_frames = 1 + n_samples // hop  →  n_samples = (n_frames - 1) * hop
                n_frames = fs[1]
                n_samples = (n_frames - 1) * base["hop_length"]
                base["duration"] = round(n_samples / base["sample_rate"], 4)

    # CLI overrides (highest priority)
    if args.sample_rate is not None: base["sample_rate"] = args.sample_rate
    if args.n_fft       is not None: base["n_fft"]       = args.n_fft
    if args.hop_length  is not None: base["hop_length"]  = args.hop_length
    if args.n_mels      is not None: base["n_mels"]      = args.n_mels
    if args.n_mfcc      is not None: base["n_mfcc"]      = args.n_mfcc
    if args.duration    is not None: base["duration"]    = args.duration

    return base


def main() -> None:
    from src.deployment.codegen import ModelToC, OnnxToC
    from src.deployment.codegen.model_to_c import BOARDS

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required
    p.add_argument("--model",  required=True, type=Path,
                   help="Path to .keras / SavedModel directory, or .onnx file")
    p.add_argument("--board",  required=True, choices=list(BOARDS),
                   help="Target board (see Available boards in help)")
    p.add_argument("--output", required=True, type=Path,
                   help="Output directory for the PlatformIO project")

    # Labels / features
    p.add_argument("--labels",       type=Path, default=None,
                   help="label_names.json (default: --features-dir/label_names.json)")
    p.add_argument("--features-dir", type=Path, default=None,
                   help="Processed feature set dir — used for label_names.json and info.json")
    p.add_argument("--class-filter", nargs="+", default=None, metavar="CLASS",
                   help="Restrict to these class names (auto-detected from optimization_report.json "
                        "when passing an ONNX from an optimization output dir)")

    # RAM budget
    p.add_argument("--max-ram", type=float, default=900.0, metavar="KB",
                   help="Maximum activation arena in KB (default: 900 — bare C on Nicla Vision)")

    # Feature extraction overrides
    p.add_argument("--sample-rate", type=int,   default=None)
    p.add_argument("--n-fft",       type=int,   default=None)
    p.add_argument("--hop-length",  type=int,   default=None)
    p.add_argument("--n-mels",      type=int,   default=None)
    p.add_argument("--n-mfcc",      type=int,   default=None)
    p.add_argument("--duration",    type=float, default=None,
                   help="Recording duration in seconds")

    args = p.parse_args()

    labels         = _load_labels(args)
    feature_params = _load_feature_params(args)

    board_cfg = BOARDS[args.board]
    print(f"\nBoard: {args.board}  "
          f"RAM={board_cfg['ram_kb']} KB  "
          f"Flash={board_cfg['flash_kb']} KB  "
          f"CPU={board_cfg['cpu']}")
    print(f"Max arena budget: {args.max_ram:.0f} KB")
    print(f"Labels ({len(labels)}): {labels}")
    print(f"Feature params: {feature_params}")

    is_onnx = args.model.suffix.lower() == ".onnx"

    if is_onnx:
        gen = OnnxToC(
            onnx_path      = args.model,
            output_dir     = args.output,
            board          = args.board,
            label_names    = labels,
            feature_params = feature_params,
            max_ram_kb     = args.max_ram,
        )
    else:
        model = _load_model(args.model)
        gen = ModelToC(
            model          = model,
            output_dir     = args.output,
            board          = args.board,
            label_names    = labels,
            feature_params = feature_params,
            max_ram_kb     = args.max_ram,
        )

    try:
        gen.generate()
    except ValueError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
