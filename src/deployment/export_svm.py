"""
Export PCA+SVM deployment bundle for Nicla Vision.

Loads a trained pca_svm joblib pipeline, extracts model parameters, and
precomputes the mel filterbank and DCT matrix needed for on-device MFCC
computation.  Writes a flat directory of .npy files + JSON configs that
nicla_main.py loads at boot.

Usage
-----
    python -m src.deployment.export_svm \\
        --model-dir  data/models/tuned/fsc22_12_classes_nicla_pca_svm_sweep \\
        --features-dir data/processed/fsc22_classical_nicla_train \\
        --train-config config/training.yaml \\
        --output deploy/nicla_svm

Deploy bundle written to <output>/
    scaler_mean.npy       (n_features,)  float32
    scaler_scale.npy      (n_features,)  float32
    pca_mean.npy          (n_features,)  float32
    pca_components.npy    (n_pca, n_features)  float32
    svm_coef.npy          (n_pairs, n_pca)  float32   — linear OvO primal weights
    svm_intercept.npy     (n_pairs,)  float32
    mel_fb.npy            (n_mels, n_bins)  float32
    dct_matrix.npy        (n_mfcc, n_mels)  float32
    freq_bins.npy         (n_bins,)  float32
    label_names.json      list[str]  — 12 class names in model order
    feature_params.json   extraction hyper-parameters for nicla_main.py

IMPORTANT — sample rate mismatch
---------------------------------
AudioClassicalExtractor default sr=22050 Hz.
Nicla Vision PDM mic supports 16 000 / 32 000 Hz ONLY.

If the model was trained at 22 050 Hz (default), on-device features will not
match training features and accuracy will degrade.  Before production deployment:

    1. Add  sample_rate: 16000  to extractor_params in feature_extraction.yaml.
    2. Re-extract features.
    3. Retune the PCA+SVM.
    4. Re-run this export script.

The export proceeds regardless so you can test the pipeline end-to-end.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import librosa
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_pipeline(model_dir: Path):
    path = model_dir / "pca_svm.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def _dct_matrix(n_mfcc: int, n_mels: int) -> np.ndarray:
    """Orthonormal DCT-II matrix matching scipy.fftpack.dct(type=2, norm='ortho').

    Returns shape (n_mfcc, n_mels) float32.
    Applied as: mfcc = dct_matrix @ log_mel_energies
    """
    k = np.arange(n_mfcc)[:, np.newaxis]          # (n_mfcc, 1)
    n = np.arange(n_mels)[np.newaxis, :]           # (1, n_mels)
    dct = np.cos(np.pi / n_mels * (n + 0.5) * k) * np.sqrt(2.0 / n_mels)
    dct[0] /= np.sqrt(2.0)                         # ortho norm for k=0
    return dct.astype(np.float32)


def _label_names(features_dir: Path, train_config: Path | None) -> list[str]:
    """Derive the 12 model-order class names.

    Reads all label names from the processed feature set, then filters and
    sorts them to match the class_filter in training.yaml (if provided).
    """
    with open(features_dir / "label_names.json") as f:
        all_names: list[str] = json.load(f)

    if train_config is None:
        print("  WARNING: --train-config not supplied; using all feature-set labels.")
        return sorted(all_names)

    with open(train_config) as f:
        cfg = yaml.safe_load(f)

    class_filter = cfg.get("class_filter")
    if not class_filter:
        return sorted(all_names)

    filter_set = set(class_filter)
    names = sorted(n for n in all_names if n in filter_set)
    if not names:
        raise ValueError(
            f"class_filter {sorted(filter_set)} produced no matches "
            f"against feature-set labels {all_names}"
        )
    return names


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------

def export(
    model_dir:         Path,
    features_dir:      Path,
    train_config:      Path | None,
    output_dir:        Path,
    extractor_params:  dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load pipeline ────────────────────────────────────────────────────────
    pipeline = _load_pipeline(model_dir)
    steps    = dict(pipeline.steps)

    scaler = steps.get("scaler")
    pca    = steps["pca"]
    svm    = steps["svm"]

    if svm.kernel != "linear":
        raise ValueError(
            f"Only linear-kernel SVM is supported for on-device deployment "
            f"(got kernel='{svm.kernel}').  Retrain with kernel='linear'."
        )

    n_classes = len(svm.classes_)
    n_pairs   = n_classes * (n_classes - 1) // 2
    n_pca     = int(pca.n_components_)

    print(f"Pipeline: scaler={'yes' if scaler else 'no'}  "
          f"PCA n_components={n_pca}  SVM n_classes={n_classes}  "
          f"n_pairs={n_pairs}")
    print(f"SVM coef_ shape: {svm.coef_.shape}  "
          f"intercept_ shape: {svm.intercept_.shape}")

    # ── Save scaler ──────────────────────────────────────────────────────────
    if scaler is not None:
        np.save(output_dir / "scaler_mean.npy",  scaler.mean_.astype(np.float32))
        np.save(output_dir / "scaler_scale.npy", scaler.scale_.astype(np.float32))
    else:
        # Identity scaler: mean=0, scale=1
        n_feat = pca.components_.shape[1]
        np.save(output_dir / "scaler_mean.npy",  np.zeros(n_feat, dtype=np.float32))
        np.save(output_dir / "scaler_scale.npy", np.ones(n_feat,  dtype=np.float32))
        print("  WARNING: no StandardScaler found — saving identity transform.")

    # ── Save PCA ─────────────────────────────────────────────────────────────
    np.save(output_dir / "pca_mean.npy",       pca.mean_.astype(np.float32))
    np.save(output_dir / "pca_components.npy", pca.components_.astype(np.float32))

    # ── Save SVM (primal form, linear kernel) ────────────────────────────────
    np.save(output_dir / "svm_coef.npy",      svm.coef_.astype(np.float32))
    np.save(output_dir / "svm_intercept.npy", svm.intercept_.astype(np.float32))

    # ── Label names ──────────────────────────────────────────────────────────
    labels = _label_names(features_dir, train_config)
    if len(labels) != n_classes:
        raise ValueError(
            f"Got {len(labels)} label names but SVM has {n_classes} classes. "
            f"Check --train-config class_filter."
        )
    with open(output_dir / "label_names.json", "w") as f:
        json.dump(labels, f, indent=2)
    print(f"  Labels: {labels}")

    # ── Feature extraction parameters ────────────────────────────────────────
    feature_params = {
        "sample_rate": extractor_params["sample_rate"],
        "n_mfcc":      extractor_params["n_mfcc"],
        "n_fft":       extractor_params["n_fft"],
        "hop_length":  extractor_params["hop_length"],
        "n_mels":      extractor_params["n_mels"],
        "duration":    extractor_params["duration"],
        "n_features":  int(pca.components_.shape[1]),
        "n_pca":       n_pca,
        "n_classes":   n_classes,
    }
    with open(output_dir / "feature_params.json", "w") as f:
        json.dump(feature_params, f, indent=2)

    sr       = feature_params["sample_rate"]
    n_fft    = feature_params["n_fft"]
    n_mels   = feature_params["n_mels"]
    n_mfcc   = feature_params["n_mfcc"]
    n_bins   = n_fft // 2 + 1

    # ── Mel filterbank ───────────────────────────────────────────────────────
    # Matches librosa.feature.mfcc internal mel filterbank exactly.
    mel_fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    np.save(output_dir / "mel_fb.npy", mel_fb.astype(np.float32))

    # ── DCT matrix ───────────────────────────────────────────────────────────
    dct = _dct_matrix(n_mfcc, n_mels)
    np.save(output_dir / "dct_matrix.npy", dct)

    # ── Frequency bin centres ────────────────────────────────────────────────
    freq_bins = np.linspace(0.0, sr / 2.0, n_bins, dtype=np.float32)
    np.save(output_dir / "freq_bins.npy", freq_bins)

    # ── Summary ──────────────────────────────────────────────────────────────
    npy_files = list(output_dir.glob("*.npy"))
    total_kb  = sum(p.stat().st_size / 1024 for p in npy_files)

    print(f"\nExported to {output_dir}/")
    for p in sorted(npy_files):
        arr = np.load(p)
        print(f"  {p.name:<28}  shape={str(arr.shape):<18}  "
              f"{p.stat().st_size/1024:6.1f} KB")
    print(f"  {'TOTAL':<28}  {total_kb:6.1f} KB")

    # Warnings
    mel_kb = mel_fb.nbytes / 1024
    if mel_kb > 100:
        print(f"\n  WARNING: mel_fb.npy is {mel_kb:.0f} KB — may exceed Nicla heap.")
        print(f"  Use --n-fft 512 --n-mels 40 to reduce to "
              f"{40*(512//2+1)*4/1024:.0f} KB (requires retraining).")

    if sr not in (16000, 32000):
        print(f"\n  WARNING: Nicla mic supports 16000/32000 Hz, got sr={sr}.")
        print(f"  Use --sample-rate 16000 (requires retraining).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir",    required=True, type=Path,
                   help="Directory containing pca_svm.joblib")
    p.add_argument("--features-dir", required=True, type=Path,
                   help="Processed feature set dir (for label_names.json)")
    p.add_argument("--train-config", default=None, type=Path,
                   help="Training YAML (for class_filter). Defaults to config/training.yaml")
    p.add_argument("--output", default="deploy/nicla_svm", type=Path,
                   help="Output directory for deploy bundle")
    # Extraction params — must match the values used during training
    p.add_argument("--sample-rate", type=int, default=16000,
                   help="Audio sample rate used during training (default: 16000)")
    p.add_argument("--n-fft",       type=int, default=512,
                   help="FFT window size used during training (default: 512)")
    p.add_argument("--hop-length",  type=int, default=160,
                   help="STFT hop length used during training (default: 160)")
    p.add_argument("--n-mfcc",      type=int, default=40,
                   help="Number of MFCC coefficients (default: 40)")
    p.add_argument("--n-mels",      type=int, default=40,
                   help="Mel filterbank bands used internally by librosa mfcc (default: 40)")
    p.add_argument("--duration",    type=float, default=3.0,
                   help="Recording duration in seconds (default: 3.0)")
    args = p.parse_args()

    if args.train_config:
        train_cfg = args.train_config
    elif Path("config/tuning.yaml").exists():
        train_cfg = Path("config/tuning.yaml")
    elif Path("config/training.yaml").exists():
        train_cfg = Path("config/training.yaml")
    else:
        train_cfg = None

    extractor_params = {
        "sample_rate": args.sample_rate,
        "n_fft":       args.n_fft,
        "hop_length":  args.hop_length,
        "n_mfcc":      args.n_mfcc,
        "n_mels":      args.n_mels,
        "duration":    args.duration,
    }

    export(args.model_dir, args.features_dir, train_cfg, args.output, extractor_params)


if __name__ == "__main__":
    main()
