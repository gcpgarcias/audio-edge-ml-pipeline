"""
tools/evaluate_device.py — evaluate a flashed model against the device test split.

The device must be flashed with EVAL_MODE 1 in main.cpp.  For each test clip
the script plays the audio through the laptop speakers in sync with the device
recording, then parses the structured inference result:

    PRED <class_name> <top_score>
    SCORES <c0> <s0> <c1> <s1> ...
    EVAL_DONE

Results are logged to MLflow with a confusion matrix PNG and classification
report.  The script adapts automatically to whatever class list the device
reports — no hard-coding of class names needed.

Usage
-----
    python tools/evaluate_device.py \\
        --manifest  data/raw/fsc22_device/split_manifest.json \\
        --source-dir data/raw/fsc22_device \\
        --experiment fsc22-device-eval \\
        --run-name   nicla-16class-trial04

    # Skip audio playback (record live sound instead):
    python tools/evaluate_device.py \\
        --manifest  data/raw/fsc22_device/split_manifest.json \\
        --experiment fsc22-device-eval
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import serial


SAMPLE_RATE = 16000
DURATION_S  = 5.0

# How long after sending 'R' before the device starts capturing (poll + warmup).
# With delay(20) READY loop and 512-sample warmup the real lag is ~50 ms.
# We start audio at the same time as 'R' — the small lag is negligible.
LEAD_IN_S = 0.0


# ── Serial helpers ─────────────────────────────────────────────────────────────

def wait_ready(ser: serial.Serial, timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    buf = ""
    while time.time() < deadline:
        chunk = ser.read(max(1, ser.in_waiting)).decode(errors="replace")
        buf += chunk
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            line = line.strip()
            if line:
                print(f"    [device] {line}")
            if line == "READY":
                return True
    return False


def recv_result(ser: serial.Serial, timeout: float = 45.0) -> dict | None:
    """Read until EVAL_DONE; return parsed prediction dict or None on error."""
    deadline = time.time() + timeout
    pred_class  = None
    pred_score  = None
    all_scores: dict[str, float] = {}

    buf = ""
    while time.time() < deadline:
        chunk = ser.read(max(1, ser.in_waiting)).decode(errors="replace")
        buf += chunk
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            if line.startswith("PRED "):
                parts = line.split()
                pred_class = parts[1]
                pred_score = float(parts[2])
            elif line.startswith("SCORES "):
                parts = line.split()[1:]   # drop "SCORES"
                for i in range(0, len(parts) - 1, 2):
                    all_scores[parts[i]] = float(parts[i + 1])
            elif line == "EVAL_DONE":
                if pred_class is not None:
                    return {"pred": pred_class, "score": pred_score, "scores": all_scores}
                print("    [warn] EVAL_DONE received but no PRED line seen")
                return None
            else:
                print(f"    [device] {line}")

    print("    ERROR: timed out waiting for EVAL_DONE")
    return None


# ── File resolution ───────────────────────────────────────────────────────────

def build_filename_index(source_dir: Path) -> dict[str, Path]:
    """Recursively index all audio files by filename for fsc22 flat layout."""
    index: dict[str, Path] = {}
    for p in source_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"}:
            index[p.name] = p
    return index


def resolve_path(
    rel: str,
    source_dir: Path,
    filename_index: dict[str, Path] | None,
) -> Path | None:
    """Return the actual audio file path for a manifest entry.

    For class-per-subfolder layout: source_dir/ClassName/file.wav
    For fsc22 flat layout (filename_index provided): look up by filename only.
    """
    if filename_index is not None:
        filename = Path(rel).name
        return filename_index.get(filename)
    p = source_dir / rel
    return p if p.exists() else None


# ── Audio helpers ──────────────────────────────────────────────────────────────

_play_thread: threading.Thread | None = None


def load_and_loop(path: Path, min_duration_s: float) -> np.ndarray:
    import librosa
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    y = y.astype(np.float32)
    min_samples = int(min_duration_s * SAMPLE_RATE)
    while len(y) < min_samples:
        y = np.concatenate([y, y])
    return y[:min_samples]


def warmup_audio() -> None:
    try:
        import sounddevice as sd
        sd.play(np.zeros(int(SAMPLE_RATE * 0.1), dtype=np.float32),
                samplerate=SAMPLE_RATE, blocking=True)
    except Exception as e:
        print(f"  [warning] audio warm-up failed: {e}")


def play_audio_start(y: np.ndarray) -> None:
    global _play_thread
    def _play():
        try:
            import sounddevice as sd
            sd.play(y, samplerate=SAMPLE_RATE, blocking=True)
        except Exception as e:
            print(f"  [ERROR] audio playback failed: {e}")
    _play_thread = threading.Thread(target=_play, daemon=True)
    _play_thread.start()


def stop_audio() -> None:
    global _play_thread
    if _play_thread is not None and _play_thread.is_alive():
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        _play_thread.join(timeout=1.0)
    _play_thread = None


# ── MLflow logging ─────────────────────────────────────────────────────────────

def log_to_mlflow(
    mlflow_uri: str,
    experiment:  str,
    run_name:    str,
    y_true:      list[str],
    y_pred:      list[str],
    all_scores:  list[dict[str, float]],
    class_names: list[str],
) -> None:
    import mlflow
    from sklearn.metrics import (
        accuracy_score, f1_score,
        classification_report, confusion_matrix,
    )
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name):
        acc        = accuracy_score(y_true, y_pred)
        f1_macro   = f1_score(y_true, y_pred, average="macro",   zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        mlflow.log_metric("accuracy",    acc)
        mlflow.log_metric("f1_macro",    f1_macro)
        mlflow.log_metric("f1_weighted", f1_weighted)
        mlflow.log_metric("n_samples",   len(y_true))

        print(f"\n{'─'*60}")
        print(f"Accuracy : {acc:.3f}")
        print(f"F1 macro : {f1_macro:.3f}")
        print(f"F1 wtd   : {f1_weighted:.3f}")
        print(f"{'─'*60}\n")

        # Classification report
        report = classification_report(y_true, y_pred, zero_division=0)
        print(report)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                         delete=False, prefix="cls_report_") as f:
            f.write(report)
            mlflow.log_artifact(f.name, artifact_path="eval")

        # Confusion matrix
        labels_present = sorted(set(y_true) | set(y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels_present)

        fig, ax = plt.subplots(figsize=(max(8, len(labels_present)),
                                        max(6, len(labels_present))))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set(xticks=range(len(labels_present)),
               yticks=range(len(labels_present)),
               xticklabels=labels_present, yticklabels=labels_present,
               ylabel="True label", xlabel="Predicted label",
               title=f"Confusion Matrix — {run_name}")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        for i in range(len(labels_present)):
            for j in range(len(labels_present)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=8)
        plt.tight_layout()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False,
                                          prefix="confusion_") as f:
            fig.savefig(f.name, dpi=150, bbox_inches="tight")
            mlflow.log_artifact(f.name, artifact_path="eval")
        plt.close(fig)

        # Per-class scores as metrics
        report_dict = classification_report(y_true, y_pred, output_dict=True,
                                            zero_division=0)
        for cls, metrics in report_dict.items():
            if isinstance(metrics, dict):
                safe = cls.replace(" ", "_")
                mlflow.log_metric(f"precision_{safe}", metrics["precision"])
                mlflow.log_metric(f"recall_{safe}",    metrics["recall"])
                mlflow.log_metric(f"f1_{safe}",        metrics["f1-score"])

        # Raw predictions as JSON artifact
        predictions = [
            {"true": t, "pred": p, "scores": s}
            for t, p, s in zip(y_true, y_pred, all_scores)
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False,
                                          prefix="predictions_") as f:
            json.dump(predictions, f, indent=2)
            mlflow.log_artifact(f.name, artifact_path="eval")

        print(f"Results logged to MLflow experiment '{experiment}' run '{run_name}'")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        prog="python tools/evaluate_device.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--manifest",    required=True,
                    help="Path to split_manifest.json")
    ap.add_argument("--source-dir",  default=None,
                    help="Root of audio dataset. If omitted, user provides sound live.")
    ap.add_argument("--loader",      default="audio_folder",
                    choices=["audio_folder", "fsc22"],
                    help="Dataset layout: audio_folder=class-per-subfolder (default), "
                         "fsc22=flat audio dir + CSV (searches by filename)")
    ap.add_argument("--split",       default="test",
                    help="Manifest split to evaluate (default: test)")
    ap.add_argument("--classes",     nargs="+", default=None,
                    help="Restrict evaluation to these class names. "
                         "Auto-populated from --optimization-report if provided.")
    ap.add_argument("--optimization-report", default=None,
                    help="Path to optimization_report.json — reads class_filter "
                         "automatically so only model classes are evaluated.")
    ap.add_argument("--port",        default=None,
                    help="Serial port (default: auto-detect /dev/cu.usbmodem*)")
    ap.add_argument("--baud",        type=int, default=115200)
    ap.add_argument("--experiment",  default="fsc22-device-eval",
                    help="MLflow experiment name")
    ap.add_argument("--run-name",    default=None,
                    help="MLflow run name (default: auto timestamp)")
    ap.add_argument("--mlflow-uri",  default="mlruns/",
                    help="MLflow tracking URI (default: mlruns/)")
    ap.add_argument("--no-mlflow",   action="store_true",
                    help="Skip MLflow logging, just print results")
    args = ap.parse_args()

    # Resolve class filter
    class_filter = None
    if args.optimization_report:
        report = json.loads(Path(args.optimization_report).read_text())
        class_filter = set(report.get("class_filter") or [])
        print(f"Class filter from optimization report ({len(class_filter)} classes): "
              f"{sorted(class_filter)}")
    elif args.classes:
        class_filter = set(args.classes)
        print(f"Class filter ({len(class_filter)} classes): {sorted(class_filter)}")

    # Load manifest
    manifest = json.loads(Path(args.manifest).read_text())
    test_files = manifest.get(args.split, [])
    if not test_files:
        print(f"ERROR: no files found under split '{args.split}' in manifest")
        sys.exit(1)

    source_dir    = Path(args.source_dir) if args.source_dir else None
    playback_mode = source_dir is not None

    # Build filename index for fsc22 flat layout
    filename_index = None
    if playback_mode and args.loader == "fsc22":
        print("Building filename index for fsc22 layout ...")
        filename_index = build_filename_index(source_dir)
        print(f"  Indexed {len(filename_index)} audio files")

    # Build list of (audio_path, true_class), filtered to model classes
    samples: list[tuple[Path | None, str]] = []
    skipped_classes: set[str] = set()
    for rel in sorted(test_files):
        class_name = rel.split("/")[0]
        if class_filter and class_name not in class_filter:
            skipped_classes.add(class_name)
            continue
        if playback_mode:
            p = resolve_path(rel, source_dir, filename_index)
            if p is None:
                print(f"  [warn] file not found, skipping: {rel}")
                continue
            samples.append((p, class_name))
        else:
            samples.append((None, class_name))

    if skipped_classes:
        print(f"Skipped {len(skipped_classes)} classes not in model: "
              f"{sorted(skipped_classes)}")
    print(f"Evaluating {len(samples)} clips from split '{args.split}'")

    # Serial port
    port = args.port
    if port is None:
        matches = sorted(_glob.glob("/dev/cu.usbmodem*"))
        if not matches:
            print("ERROR: no /dev/cu.usbmodem* device found.")
            sys.exit(1)
        port = matches[0]
        if len(matches) > 1:
            print(f"Multiple ports {matches} — using {port}")

    ser = serial.Serial(port, args.baud, timeout=1)
    print(f"Opened {port} @ {args.baud} baud")

    if playback_mode:
        print("Warming up audio system ...")
        warmup_audio()
        print("Audio ready.\n")

    y_true:     list[str]             = []
    y_pred:     list[str]             = []
    all_scores: list[dict[str, float]] = []
    errors = 0

    try:
        for i, (audio_path, true_class) in enumerate(samples):
            print(f"─── {i+1}/{len(samples)} — true: {true_class} ───")
            if audio_path:
                print(f"  Source: {audio_path.name}")

            # Pre-load audio
            clip_audio = None
            if playback_mode and audio_path:
                clip_audio = load_and_loop(audio_path, DURATION_S)

            # Flush stale serial data and wait for READY
            ser.reset_input_buffer()
            print("  Waiting for READY ...")
            if not wait_ready(ser, timeout=15.0):
                print("  ERROR: device did not send READY — is EVAL_MODE=1?")
                errors += 1
                continue

            # Trigger + audio simultaneously
            ser.write(b'R')
            ser.flush()
            if playback_mode and clip_audio is not None:
                play_audio_start(clip_audio)
                print(f"  Triggered + audio playing — waiting for inference ...")
            else:
                print(f"  Triggered — make the sound '{true_class}' now ...")

            # Wait for structured result
            result = recv_result(ser, timeout=45.0)
            stop_audio()

            if result is None:
                print("  ERROR: no result received — skipping")
                errors += 1
                continue

            pred_class = result["pred"]
            correct    = "✓" if pred_class == true_class else "✗"
            print(f"  {correct} pred={pred_class} (score={result['score']:.3f})")

            y_true.append(true_class)
            y_pred.append(pred_class)
            all_scores.append(result["scores"])

    except KeyboardInterrupt:
        print(f"\nStopped after {len(y_true)}/{len(samples)} clips.")
    finally:
        ser.close()

    if not y_true:
        print("No results collected.")
        sys.exit(1)

    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    print(f"\nCollected {len(y_true)} results  |  {errors} errors")
    print(f"Quick accuracy: {acc:.3f}")

    run_name = args.run_name or f"device_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    class_names = sorted(set(y_true))

    if not args.no_mlflow:
        log_to_mlflow(
            mlflow_uri=args.mlflow_uri,
            experiment=args.experiment,
            run_name=run_name,
            y_true=y_true,
            y_pred=y_pred,
            all_scores=all_scores,
            class_names=class_names,
        )
    else:
        from sklearn.metrics import classification_report
        print(classification_report(y_true, y_pred, zero_division=0))


if __name__ == "__main__":
    main()