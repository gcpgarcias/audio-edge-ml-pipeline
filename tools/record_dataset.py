"""
tools/record_dataset.py — record a labelled device dataset via the Nicla PDM mic.

The device must be flashed with PCM_DUMP_MODE 1 in main.cpp.  The tool sends
an 'R' trigger over serial, which makes the device start recording immediately.
If --source-dir is given, a random clip from that class is played through the
laptop speakers in sync with the recording.  Otherwise the user provides the
sound live (a countdown is printed).

Output layout (compatible with AudioFolderLoader)::

    <output_dir>/
        <ClassName>/
            device_<timestamp>.wav
            ...

Usage
-----
    # Play clips from fsc22 through speakers while device records:
    python tools/record_dataset.py \\
        --class Chainsaw \\
        --n 30 \\
        --source-dir data/raw/fsc22 \\
        --output data/raw/fsc22_device

    # Record live sound (user provides the sound):
    python tools/record_dataset.py \\
        --class Silence \\
        --n 20 \\
        --output data/raw/fsc22_device

After recording all classes, generate the train/val/test split manifest::

    python tools/generate_split.py --input data/raw/fsc22_device
"""

from __future__ import annotations

import argparse
import glob as _glob
import random
import threading
import struct
import sys
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import serial
import soundfile as sf

MAGIC_START  = bytes([0xCA, 0xFE, 0xBA, 0xBE])
MAGIC_END    = bytes([0xDE, 0xAD, 0xBE, 0xEF])
SAMPLE_RATE  = 16000
DURATION_S   = 5.0
N_SAMPLES    = int(SAMPLE_RATE * DURATION_S)

# How long to play audio BEFORE sending 'R' to the device.
# The OS audio pipeline needs time to actually push samples to the speakers
# after sd.play() is called — observed latency on macOS is ~2 s on the first
# call.  Audio is started first, we sleep this long, then trigger recording.
# Increase if the start of clips is still cut off; decrease to taste.
LEAD_IN_S = 0.5   # after audio warm-up, sounddevice latency is < 100 ms


# ── Serial helpers ────────────────────────────────────────────────────────────

def wait_ready(ser: serial.Serial, timeout: float = 10.0) -> bool:
    """Wait for 'READY\n' from the device (printed before it waits for 'R')."""
    deadline = time.time() + timeout
    buf = ""
    while time.time() < deadline:
        chunk = ser.read(max(1, ser.in_waiting)).decode(errors="replace")
        buf += chunk
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            line = line.strip()
            if line:
                print(f"  [device] {line}")
            if line == "READY":
                return True
    return False


def recv_pcm(ser: serial.Serial) -> bytes | None:
    """Read one binary PCM frame from serial; return raw int16 bytes or None on error."""
    buf = b""
    while True:
        buf += ser.read(max(1, ser.in_waiting))

        idx = buf.find(MAGIC_START)
        if idx == -1:
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                text = line.decode(errors="replace").strip()
                if text:
                    print(f"  [device] {text}")
                if text == "TIMEOUT":
                    print("  ERROR: device reported timeout")
                    return None
            buf = buf[-8:]
            continue

        if idx > 0:
            print(f"  [device] {buf[:idx].decode(errors='replace').strip()}")
        buf = buf[idx + 4:]

        while len(buf) < 4:
            buf += ser.read(4 - len(buf))
        n_samples = struct.unpack_from("<I", buf)[0]
        buf = buf[4:]

        n_bytes = n_samples * 2
        while len(buf) < n_bytes + 4:
            chunk = ser.read(min(4096, n_bytes + 4 - len(buf)))
            if not chunk:
                break
            buf += chunk

        if len(buf) < n_bytes + 4:
            print("  ERROR: incomplete frame")
            return None

        payload  = buf[:n_bytes]
        end_mark = buf[n_bytes:n_bytes + 4]
        if end_mark != MAGIC_END:
            print(f"  WARNING: bad end marker {end_mark.hex()}")
        return payload


# ── Audio helpers ─────────────────────────────────────────────────────────────

def load_and_loop(path: Path, min_duration_s: float) -> np.ndarray:
    """Load audio, resample, loop until at least min_duration_s of audio is available.

    Looping ensures the recording window is always filled with sound regardless
    of when within the playback the device starts capturing.
    """
    import librosa
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    y = y.astype(np.float32)
    min_samples = int(min_duration_s * SAMPLE_RATE)
    while len(y) < min_samples:
        y = np.concatenate([y, y])
    return y[:min_samples]


def warmup_audio() -> None:
    """Play a short silent buffer to initialise the CoreAudio session.

    The first sd.play() call on macOS cold-starts the CoreAudio pipeline
    which can take 2-5 seconds.  This call absorbs that cost before the
    recording loop so every subsequent play starts in < 100 ms.
    """
    try:
        import sounddevice as sd
        sd.play(np.zeros(int(SAMPLE_RATE * 0.1), dtype=np.float32),
                samplerate=SAMPLE_RATE, blocking=True)
    except Exception as e:
        print(f"  [warning] audio warm-up failed: {e}")


_play_thread: threading.Thread | None = None


def play_audio_start(y: np.ndarray, sr: int) -> None:
    """Start audio playback in a background thread. Errors are printed, not swallowed."""
    global _play_thread

    def _play() -> None:
        try:
            import sounddevice as sd
            sd.play(y, samplerate=sr, blocking=True)
        except Exception as e:
            print(f"  [ERROR] audio playback failed: {e}")

    _play_thread = threading.Thread(target=_play, daemon=True)
    _play_thread.start()


def stop_audio(_proc=None) -> None:
    global _play_thread
    if _play_thread is not None and _play_thread.is_alive():
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        _play_thread.join(timeout=1.0)
    _play_thread = None


def resolve_candidates(source_dir: Path, class_name: str) -> list[Path]:
    """Return all audio files for class_name under source_dir.

    Supports two layouts:
    - class-per-subfolder:  source_dir/<ClassName>/*.wav
    - fsc22 flat + CSV:     source_dir/**/Metadata*.csv maps filenames → class names
    """
    # Try class-per-subfolder first
    class_dir = source_dir / class_name
    if class_dir.is_dir():
        exts = {".wav", ".flac", ".mp3", ".ogg"}
        return [f for f in sorted(class_dir.iterdir()) if f.suffix.lower() in exts]

    # Fall back to fsc22-style: find a Metadata CSV and the flat audio directory
    import csv
    csv_files = list(source_dir.rglob("*.csv"))
    audio_dirs = [p for p in source_dir.rglob("*") if p.is_dir()
                  and any(f.suffix.lower() == ".wav" for f in p.iterdir()
                          if f.is_file())]
    if not csv_files or not audio_dirs:
        print(f"  [warning] no class folder or metadata CSV found under {source_dir}")
        return []

    csv_path  = csv_files[0]
    audio_dir = audio_dirs[0]
    candidates = []
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # CSV columns: Source File Name, Dataset File Name, Class ID, Class Name
            if row.get("Class Name", "").strip() == class_name:
                dataset_fname = row.get("Dataset File Name", "").strip()
                path = audio_dir / dataset_fname
                if path.exists():
                    candidates.append(path)
    if not candidates:
        print(f"  [warning] class '{class_name}' not found in {csv_path.name}")
    return candidates


def build_clip_queue(candidates: list[Path], n: int, rng: random.Random) -> list[Path]:
    """Return a list of length n drawn from candidates without repeating within a pass."""
    if not candidates:
        return []
    queue: list[Path] = []
    while len(queue) < n:
        pass_ = list(candidates)
        rng.shuffle(pass_)
        queue.extend(pass_)
    return queue[:n]


def save_wav(pcm_bytes: bytes, path: Path) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        prog="python tools/record_dataset.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--class",      dest="class_name", required=True,
                    help="Class label to record (e.g. Chainsaw)")
    ap.add_argument("--n",          type=int, default=30,
                    help="Number of recordings to capture (default: 30)")
    ap.add_argument("--output",     default="data/raw/fsc22_device",
                    help="Root output directory (default: data/raw/fsc22_device)")
    ap.add_argument("--source-dir", default=None,
                    help="Root of source dataset to play clips from "
                         "(e.g. data/raw/fsc22).  If omitted, user provides sound live.")
    ap.add_argument("--port",       default=None,
                    help="Serial port (default: auto-detect /dev/cu.usbmodem*)")
    ap.add_argument("--baud",       type=int, default=115200)
    ap.add_argument("--seed",       type=int, default=None,
                    help="RNG seed for clip selection (default: random)")
    args = ap.parse_args()

    class_name = args.class_name
    output_dir = Path(args.output) / class_name
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(args.source_dir) if args.source_dir else None
    rng = random.Random(args.seed)

    playback_mode = source_dir is not None
    if playback_mode:
        print(f"Playback mode: clips from {source_dir} (class: {class_name})")
        candidates = resolve_candidates(source_dir, class_name)
        clip_queue = build_clip_queue(candidates, args.n, rng)
        if not clip_queue:
            print("ERROR: no source clips found — aborting.")
            raise SystemExit(1)
        print(f"  {len(set(clip_queue))} unique clips, cycling across {args.n} recordings\n")
    else:
        clip_queue = []
        print("Live mode: make the sound yourself when prompted.")

    print(f"Recording {args.n} clips of '{class_name}' → {output_dir}\n")

    port = args.port
    if port is None:
        matches = sorted(_glob.glob("/dev/cu.usbmodem*"))
        if not matches:
            print("ERROR: no /dev/cu.usbmodem* device found. Is the Nicla plugged in?")
            raise SystemExit(1)
        port = matches[0]
        if len(matches) > 1:
            print(f"Multiple ports found {matches} — using {port}")

    ser = serial.Serial(port, args.baud, timeout=1)
    print(f"Opened {port} @ {args.baud} baud")

    if playback_mode:
        print("Warming up audio system (first CoreAudio init) ...")
        warmup_audio()
        print("Audio ready.\n")

    captured = 0
    try:
        while captured < args.n:
            print(f"─── Clip {captured + 1}/{args.n} ───")

            # Load clip (looped to cover lead-in + recording window)
            clip_audio = None
            if playback_mode:
                clip_path = clip_queue[captured]
                clip_audio = load_and_loop(clip_path, DURATION_S)
                print(f"  Source: {clip_path.name}")

            # Flush stale serial data before waiting (guards against previous timeout)
            ser.reset_input_buffer()
            # Wait for device to print READY
            print("  Waiting for device READY ...")
            if not wait_ready(ser, timeout=15.0):
                print("  ERROR: device did not send READY — is PCM_DUMP_MODE=1?")
                break

            audio_proc = None
            if playback_mode:
                # Trigger first so device warmup (~32 ms) overlaps with CoreAudio
                # startup (~10 ms after warmup_audio).  Audio starts immediately
                # after the trigger — both begin within ~50 ms of each other.
                ser.write(b'R')
                ser.flush()
                print(f"  Triggered — device recording {DURATION_S:.0f}s ...")
                audio_proc = play_audio_start(clip_audio, SAMPLE_RATE)
            else:
                print(f"  Get ready to make the sound '{class_name}' ...")
                for i in (3, 2, 1):
                    print(f"  {i}...")
                    time.sleep(1.0)
                ser.write(b'R')
                ser.flush()
                print(f"  Triggered — device recording {DURATION_S:.0f}s ...")

            # Receive PCM from device
            pcm_bytes = recv_pcm(ser)
            stop_audio(audio_proc)
            if pcm_bytes is None:
                print("  Skipping this clip due to receive error.\n")
                continue

            # Save
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = output_dir / f"device_{ts}.wav"
            save_wav(pcm_bytes, out_path)
            n_received = len(pcm_bytes) // 2
            print(f"  Saved {out_path.name}  ({n_received} samples, "
                  f"{n_received / SAMPLE_RATE:.1f}s)\n")
            captured += 1

    except KeyboardInterrupt:
        print(f"\nStopped after {captured}/{args.n} clips.")
    finally:
        ser.close()

    print(f"\nDone. {captured} clips saved to {output_dir}")
    if captured > 0:
        print("Next step: python tools/generate_split.py --input data/raw/fsc22_device")


if __name__ == "__main__":
    main()