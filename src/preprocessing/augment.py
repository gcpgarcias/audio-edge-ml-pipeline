"""
Audio data augmentation — Stage 1b.

Reads a dataset's training split (via FSC22Loader or AudioFolderLoader),
produces N augmented copies of each file, and writes the results to an
output directory in class-per-subfolder layout suitable for AudioFolderLoader.

Original files are copied alongside their augmented counterparts so the
output directory is a self-contained, drop-in training set.

Usage
-----
    python -m src.preprocessing.augment --config config/augmentation.yaml

Output layout::

    <output_dir>/
        ClassName/
            original_file.wav
            original_file_aug001.wav
            original_file_aug002.wav
            ...

The output is consumed by Stage 2 via the audio_folder loader::

    python -m src.preprocessing.pipeline \\
        --loader audio_folder \\
        --audio-folder <output_dir> \\
        --extractor audio_classical \\
        --output data/processed/<run_name>

Augmentation functions
----------------------
volume_scale      Multiply waveform by a random gain in [min_gain, max_gain].
gaussian_noise    Add zero-mean Gaussian noise scaled to [min_amplitude, max_amplitude].
time_stretch      Stretch or compress the time axis by a random rate in [min_rate, max_rate].
pitch_shift       Shift pitch by a random number of semitones in [min_steps, max_steps].
time_shift        Cyclically roll the waveform by up to max_fraction × length samples.
polarity_inversion  Flip the sign of the waveform (no audible change for most classifiers).

Each augmented copy applies ALL enabled augmentations in sequence with
independently re-sampled parameters, producing compound variation.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Augmentation functions
# Each signature: (y, sr, rng, **params) -> np.ndarray
# ---------------------------------------------------------------------------

def _volume_scale(
    y: np.ndarray, sr: int, rng: np.random.Generator,
    min_gain: float = 0.7, max_gain: float = 1.3,
) -> np.ndarray:
    gain = rng.uniform(min_gain, max_gain)
    return (y * gain).astype(y.dtype)


def _gaussian_noise(
    y: np.ndarray, sr: int, rng: np.random.Generator,
    min_amplitude: float = 0.001, max_amplitude: float = 0.008,
) -> np.ndarray:
    amplitude = rng.uniform(min_amplitude, max_amplitude)
    noise = rng.standard_normal(len(y)).astype(y.dtype) * amplitude
    return np.clip(y + noise, -1.0, 1.0).astype(y.dtype)


def _time_stretch(
    y: np.ndarray, sr: int, rng: np.random.Generator,
    min_rate: float = 0.85, max_rate: float = 1.15,
) -> np.ndarray:
    rate = rng.uniform(min_rate, max_rate)
    return librosa.effects.time_stretch(y, rate=rate)


def _pitch_shift(
    y: np.ndarray, sr: int, rng: np.random.Generator,
    min_steps: float = -3.0, max_steps: float = 3.0,
) -> np.ndarray:
    n_steps = rng.uniform(min_steps, max_steps)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def _time_shift(
    y: np.ndarray, sr: int, rng: np.random.Generator,
    max_fraction: float = 0.2,
) -> np.ndarray:
    shift = int(rng.uniform(-max_fraction, max_fraction) * len(y))
    return np.roll(y, shift).astype(y.dtype)


def _polarity_inversion(
    y: np.ndarray, sr: int, rng: np.random.Generator,
) -> np.ndarray:
    return (-y).astype(y.dtype)


_AUGMENTORS: dict[str, callable] = {
    "volume_scale":       _volume_scale,
    "gaussian_noise":     _gaussian_noise,
    "time_stretch":       _time_stretch,
    "pitch_shift":        _pitch_shift,
    "time_shift":         _time_shift,
    "polarity_inversion": _polarity_inversion,
}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _apply_augmentations(
    y: np.ndarray,
    sr: int,
    aug_specs: list[dict],
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply all augmentations in *aug_specs* sequentially to *y*."""
    y_out = y.copy()
    for spec in aug_specs:
        aug_type = spec["type"]
        if aug_type not in _AUGMENTORS:
            raise ValueError(
                f"Unknown augmentation type '{aug_type}'. "
                f"Valid types: {sorted(_AUGMENTORS)}"
            )
        params = {k: v for k, v in spec.items() if k != "type"}
        y_out = _AUGMENTORS[aug_type](y_out, sr, rng, **params)
    return y_out


def _preserve_length(y_aug: np.ndarray, original_length: int) -> np.ndarray:
    """Trim or zero-pad *y_aug* to match *original_length*."""
    if len(y_aug) > original_length:
        return y_aug[:original_length]
    if len(y_aug) < original_length:
        return np.pad(y_aug, (0, original_length - len(y_aug)))
    return y_aug


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with path.open() as fh:
        cfg = yaml.safe_load(fh) or {}

    # Validate required fields
    for key in ("output_dir",):
        if key not in cfg:
            raise ValueError(f"augmentation.yaml must include '{key}'.")

    cfg.setdefault("n_augments", 4)
    cfg.setdefault("preserve_length", True)
    cfg.setdefault("seed", 42)
    cfg.setdefault("sample_rate", None)   # None = keep original sr
    cfg.setdefault("augmentations", [])
    cfg.setdefault("class_overrides", {})
    cfg.setdefault("loader", "audio_folder")
    cfg.setdefault("split", "train")
    return cfg


# ---------------------------------------------------------------------------
# Dataset iteration helpers
# ---------------------------------------------------------------------------

def _iter_fsc22(cfg: dict):
    """Yield (audio_path, class_name) for the FSC22 train split."""
    from src.preprocessing.dataset_loaders.fsc22_loader import FSC22Loader

    dataset_root = cfg.get("dataset")
    if not dataset_root:
        raise ValueError("augmentation.yaml must include 'dataset' when loader=fsc22.")

    loader = FSC22Loader(
        dataset_root = Path(dataset_root),
        split        = cfg.get("split", "train"),
        seed         = cfg.get("seed", 42),
    )
    for audio_path, class_name, _ in loader:
        yield audio_path, class_name


def _iter_audio_folder(cfg: dict):
    """Yield (audio_path, class_name) for a class-per-subfolder directory."""
    audio_folder = cfg.get("audio_folder") or cfg.get("dataset")
    if not audio_folder:
        raise ValueError(
            "augmentation.yaml must include 'audio_folder' when loader=audio_folder."
        )
    root = Path(audio_folder)
    extensions = {".wav", ".flac", ".mp3", ".ogg", ".aiff"}
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() in extensions:
                yield f, class_dir.name


_LOADERS = {
    "fsc22":        _iter_fsc22,
    "audio_folder": _iter_audio_folder,
}


# ---------------------------------------------------------------------------
# Main augmentation routine
# ---------------------------------------------------------------------------

def run(cfg: dict) -> None:
    output_dir    = Path(cfg["output_dir"])
    n_augments    = int(cfg["n_augments"])
    preserve_len  = bool(cfg["preserve_length"])
    seed          = int(cfg["seed"])
    target_sr     = cfg["sample_rate"]
    default_augs  = cfg["augmentations"]
    class_overrides = cfg["class_overrides"]
    loader_name   = cfg["loader"]

    if loader_name not in _LOADERS:
        raise ValueError(
            f"Unknown loader '{loader_name}'. Valid: {sorted(_LOADERS)}"
        )

    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all (path, class_name) pairs first for progress logging
    logger.info("Scanning dataset (loader=%s) ...", loader_name)
    samples = list(_LOADERS[loader_name](cfg))
    logger.info("Found %d files across %d classes.",
                len(samples),
                len({c for _, c in samples}))

    # Group by class for cleaner progress reporting
    by_class: dict[str, list[Path]] = {}
    for path, class_name in samples:
        by_class.setdefault(class_name, []).append(path)

    total_written = 0
    for class_name, paths in sorted(by_class.items()):
        class_dir = output_dir / class_name
        class_dir.mkdir(exist_ok=True)

        aug_specs = class_overrides.get(class_name, {}).get(
            "augmentations", default_augs
        )

        n_orig = len(paths)
        for audio_path in paths:
            # Load audio
            y, sr = librosa.load(
                audio_path,
                sr=target_sr,
                mono=True,
            )
            original_length = len(y)

            # Copy original
            dest_orig = class_dir / audio_path.name
            if not dest_orig.exists():
                shutil.copy2(audio_path, dest_orig)

            # Write N augmented copies
            for i in range(1, n_augments + 1):
                y_aug = _apply_augmentations(y, sr, aug_specs, rng)
                if preserve_len:
                    y_aug = _preserve_length(y_aug, original_length)
                out_name = f"{audio_path.stem}_aug{i:03d}.wav"
                sf.write(class_dir / out_name, y_aug, sr)
                total_written += 1

        logger.info(
            "  %-20s  %d orig → %d total (%d augmented)",
            class_name,
            n_orig,
            n_orig * (1 + n_augments),
            n_orig * n_augments,
        )

    logger.info(
        "Done. Wrote %d augmented files to %s  (originals copied alongside).",
        total_written,
        output_dir,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.preprocessing.augment",
        description="Stage 1b — Audio data augmentation",
    )
    parser.add_argument(
        "--config", metavar="YAML", required=True,
        help="Path to augmentation config YAML.",
    )
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logger.error("Config not found: %s", cfg_path)
        sys.exit(1)

    cfg = load_config(cfg_path)
    logger.info("Augmentation config: n_augments=%d  preserve_length=%s  seed=%d",
                cfg["n_augments"], cfg["preserve_length"], cfg["seed"])
    run(cfg)


if __name__ == "__main__":
    main()
