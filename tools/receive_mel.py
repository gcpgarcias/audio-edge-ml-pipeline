"""
tools/receive_mel.py — receive a mel spectrogram dumped by the Nicla and
compare it side-by-side with the closest training sample.

Usage
-----
    python tools/receive_mel.py \
        --port /dev/cu.usbmodem11201 \
        --features data/processed/fsc22_melspec_val_2 \
        --experiment fsc22-nicla-3-classes \
        --label Speaking          # optional: filter training samples by class

The device must be flashed with FEAT_DUMP_MODE 1 in main.cpp.

Captured .npy files are saved under data/debug/<experiment>/mel/.
"""

import argparse
import glob
import struct
import sys
from datetime import datetime
import numpy as np
import serial
import json
import matplotlib.pyplot as plt
from pathlib import Path

MAGIC_START = bytes([0xFE, 0xED, 0x12, 0x34])
MAGIC_END   = bytes([0xDE, 0xAD, 0x56, 0x78])


# ── Serial receiver ──────────────────────────────────────────────────────────

def recv_frame(ser: serial.Serial) -> np.ndarray:
    """Block until one binary mel frame arrives; return float32 array."""
    print("Waiting for device... (trigger a recording)")
    buf = b""
    while True:
        buf += ser.read(max(1, ser.in_waiting))
        idx = buf.find(MAGIC_START)
        if idx == -1:
            # Print any text lines for visibility
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                try:
                    text = line.decode(errors="replace").strip()
                    if text:
                        print("[device]", text)
                    if text == "READY":
                        ser.write(b'R')
                        ser.flush()
                        print("[host] sent trigger 'R'")
                except Exception:
                    pass
            buf = buf[-4:]          # keep tail in case magic spans reads
            continue
        buf = buf[idx + 4:]         # consume magic_start

        # Read n_floats (4 bytes)
        while len(buf) < 4:
            buf += ser.read(4 - len(buf))
        n_floats = struct.unpack_from("<I", buf)[0]
        buf = buf[4:]

        n_bytes = n_floats * 4
        print(f"Receiving {n_floats} floats ({n_bytes / 1024:.1f} KB)…")
        while len(buf) < n_bytes + 4:
            chunk = ser.read(min(4096, n_bytes + 4 - len(buf)))
            buf += chunk
            pct = min(100, 100 * len(buf) / (n_bytes + 4))
            print(f"  {pct:.0f}%", end="\r")
        print()

        payload  = buf[:n_bytes]
        end_mark = buf[n_bytes:n_bytes + 4]
        if end_mark != MAGIC_END:
            print(f"WARNING: bad end marker {end_mark.hex()} — frame may be corrupt")

        floats = np.frombuffer(payload, dtype=np.float32).copy()
        return floats


# ── Training set loader ──────────────────────────────────────────────────────

def load_training_samples(features_dir: Path, label_filter: str | None):
    features   = np.load(features_dir / "features.npy")   # (N, H, W)
    labels_int = np.load(features_dir / "labels.npy")
    label_names = json.load(open(features_dir / "label_names.json"))

    if label_filter:
        if label_filter not in label_names:
            sys.exit(f"Label '{label_filter}' not found. Available: {label_names}")
        idx = label_names.index(label_filter)
        mask = labels_int == idx
        features   = features[mask]
        labels_int = labels_int[mask]
        print(f"Loaded {len(features)} '{label_filter}' training samples")
    else:
        print(f"Loaded {len(features)} training samples ({len(label_names)} classes)")

    return features, labels_int, label_names


# ── Comparison ───────────────────────────────────────────────────────────────

def find_closest(device_feat: np.ndarray, train_features: np.ndarray):
    """Return index of the training sample with minimum MSE to device_feat."""
    flat = train_features.reshape(len(train_features), -1)
    mse  = np.mean((flat - device_feat.reshape(1, -1)) ** 2, axis=1)
    return int(np.argmin(mse)), float(mse.min())


def plot_comparison(device_feat: np.ndarray, train_feat: np.ndarray,
                    device_shape: tuple, label: str, mse: float,
                    save_path: Path | None):
    device_2d = device_feat.reshape(device_shape)
    train_2d  = train_feat.reshape(device_shape) if train_feat.shape != device_shape else train_feat

    vmin = min(device_2d.min(), train_2d.min())
    vmax = max(device_2d.max(), train_2d.max())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].imshow(device_2d, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    axes[0].set_title("Device (Nicla mic)")
    axes[0].set_xlabel("Frame"); axes[0].set_ylabel("Mel bin")

    axes[1].imshow(train_2d, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Closest training sample ({label})\nMSE={mse:.4f}")
    axes[1].set_xlabel("Frame")

    diff = device_2d - train_2d
    im = axes[2].imshow(diff, aspect="auto", origin="lower",
                        cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    axes[2].set_title("Difference (device − training)")
    axes[2].set_xlabel("Frame")
    fig.colorbar(im, ax=axes[2])

    # Distribution comparison
    fig2, ax = plt.subplots(figsize=(8, 4))
    ax.hist(device_2d.ravel(), bins=80, alpha=0.6, label="Device", density=True)
    ax.hist(train_2d.ravel(),  bins=80, alpha=0.6, label="Training", density=True)
    ax.set_xlabel("Feature value"); ax.set_ylabel("Density")
    ax.set_title("Feature value distributions")
    ax.legend()

    print(f"\nDevice  — min={device_2d.min():.4f}  max={device_2d.max():.4f}"
          f"  mean={device_2d.mean():.4f}  std={device_2d.std():.4f}")
    print(f"Training— min={train_2d.min():.4f}  max={train_2d.max():.4f}"
          f"  mean={train_2d.mean():.4f}  std={train_2d.std():.4f}")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        fig2.savefig(save_path.with_suffix(".dist.png"), dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.tight_layout()
    plt.show()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port",       default=None,
                    help="Serial port (default: auto-detect /dev/cu.usbmodem*)")
    ap.add_argument("--baud",       type=int, default=115200)
    ap.add_argument("--features",   default="data/processed/fsc22_melspec_val_2",
                    help="Path to processed feature set directory")
    ap.add_argument("--experiment", default="default",
                    help="Experiment name — determines output directory "
                         "data/debug/<experiment>/mel/")
    ap.add_argument("--label",      default=None,
                    help="Filter training samples to this class label")
    ap.add_argument("--save",       default=None,
                    help="Save comparison plot to this PNG path")
    ap.add_argument("--no-device",  action="store_true",
                    help="Load a saved .npy instead of reading from serial (--load)")
    ap.add_argument("--load",       default=None,
                    help="Load device features from .npy file instead of serial")
    args = ap.parse_args()

    features_dir = Path(args.features)

    # Load device features
    if args.load:
        device_feat = np.load(args.load)
        print(f"Loaded device features from {args.load}: shape={device_feat.shape}")
    else:
        port = args.port
        if port is None:
            matches = sorted(glob.glob("/dev/cu.usbmodem*"))
            if not matches:
                print("ERROR: no /dev/cu.usbmodem* device found. Is the Nicla plugged in?")
                sys.exit(1)
            port = matches[0]
            if len(matches) > 1:
                print(f"Multiple ports found {matches} — using {port}")
        ser = serial.Serial(port, args.baud, timeout=30)
        print(f"Opened {port} @ {args.baud}")
        device_feat = recv_frame(ser)
        ser.close()
        # Save for offline reuse under data/debug/<experiment>/mel/
        out_dir = Path("data/debug") / args.experiment / "mel"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_npy = out_dir / f"device_mel_{ts}.npy"
        np.save(save_npy, device_feat)
        print(f"Saved raw device features to {save_npy}")

    # Load training samples
    train_features, train_labels, label_names = load_training_samples(
        features_dir, args.label
    )

    # Infer spectrogram shape from training data
    spec_shape = train_features.shape[1:]   # (H, W) = (n_mels, n_frames)
    n_expected = spec_shape[0] * spec_shape[1]

    if len(device_feat) != n_expected:
        print(f"WARNING: device sent {len(device_feat)} floats but training "
              f"shape is {spec_shape} ({n_expected} floats)")
        # Trim or pad to match
        if len(device_feat) > n_expected:
            print(f"  Trimming device features to {n_expected}")
            device_feat = device_feat[:n_expected]
        else:
            print(f"  Zero-padding device features to {n_expected}")
            device_feat = np.pad(device_feat, (0, n_expected - len(device_feat)))

    # Find closest training sample
    closest_idx, mse = find_closest(device_feat, train_features)
    closest_label = label_names[train_labels[closest_idx]]
    print(f"\nClosest training sample: index={closest_idx}  "
          f"label='{closest_label}'  MSE={mse:.4f}")

    save_path = Path(args.save) if args.save else None
    plot_comparison(device_feat, train_features[closest_idx],
                    spec_shape, closest_label, mse, save_path)


if __name__ == "__main__":
    main()
