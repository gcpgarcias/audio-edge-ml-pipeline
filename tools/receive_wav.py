"""
tools/receive_wav.py — receive raw PCM from the Nicla and save as a .wav file.

The device must be flashed with PCM_DUMP_MODE 1 in main.cpp.
PCM is captured after DC removal and the 4 kHz notch filter, so what you
hear is exactly what the feature extractor sees.

Usage
-----
    python tools/receive_wav.py --port /dev/cu.usbmodem11201 --experiment fsc22-nicla-3-classes
    python tools/receive_wav.py --port /dev/cu.usbmodem11201 --out recording.wav

Controls
--------
  - Each recording is saved under data/debug/<experiment>/wav/ with a timestamp.
  - Press Ctrl-C to stop after the current recording.
"""

import argparse
import glob
import struct
import sys
import wave
from datetime import datetime
from pathlib import Path

import serial

MAGIC_START = bytes([0xCA, 0xFE, 0xBA, 0xBE])
MAGIC_END   = bytes([0xDE, 0xAD, 0xBE, 0xEF])
SAMPLE_RATE = 16000


def recv_pcm(ser: serial.Serial) -> bytes:
    """Block until one binary PCM frame arrives; return raw int16 bytes."""
    print("Waiting for device... (trigger a recording)")
    buf = b""
    while True:
        buf += ser.read(max(1, ser.in_waiting))

        idx = buf.find(MAGIC_START)
        if idx == -1:
            # Print any complete text lines, keep tail in case magic spans reads
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
            buf = buf[-8:]  # keep enough tail for magic to span reads
            continue

        # Print any text before the magic marker
        if idx > 0:
            try:
                print("[device]", buf[:idx].decode(errors="replace").strip())
            except Exception:
                pass
        buf = buf[idx + 4:]

        # Read n_samples (4 bytes, uint32 little-endian)
        while len(buf) < 4:
            buf += ser.read(4 - len(buf))
        n_samples = struct.unpack_from("<I", buf)[0]
        buf = buf[4:]

        n_bytes = n_samples * 2  # int16 = 2 bytes per sample
        duration_s = n_samples / SAMPLE_RATE
        print(f"Receiving {n_samples} samples ({duration_s:.1f}s, {n_bytes/1024:.1f} KB)…")

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

        return payload


def save_wav(pcm_bytes: bytes, path: Path):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)   # int16
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)
    print(f"Saved {path}  ({len(pcm_bytes)//2} samples, "
          f"{len(pcm_bytes)//(2*SAMPLE_RATE):.1f}s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port",       default=None,
                    help="Serial port (default: auto-detect /dev/cu.usbmodem*)")
    ap.add_argument("--baud",       type=int, default=115200)
    ap.add_argument("--out",        default=None,
                    help="Output .wav path. If omitted, auto-named with timestamp.")
    ap.add_argument("--experiment", default="default",
                    help="Experiment name — determines output directory "
                         "data/debug/<experiment>/wav/")
    ap.add_argument("--count",      type=int, default=1,
                    help="Number of recordings to capture (default: 1, 0 = loop forever)")
    args = ap.parse_args()

    out_dir = Path("data/debug") / args.experiment / "wav"
    out_dir.mkdir(parents=True, exist_ok=True)

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

    i = 0
    try:
        while args.count == 0 or i < args.count:
            pcm_bytes = recv_pcm(ser)

            if args.out and args.count == 1:
                out_path = Path(args.out)
            else:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = out_dir / f"device_recording_{ts}.wav"

            save_wav(pcm_bytes, out_path)
            i += 1
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
