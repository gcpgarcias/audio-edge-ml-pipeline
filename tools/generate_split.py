"""
tools/generate_split.py — generate a reproducible train/val/test split manifest.

Supports two dataset layouts:

  class-per-subfolder (default):
      <root>/<ClassName>/<file.wav>
      Used by: fsc22_device, any AudioFolderLoader dataset.

  fsc22 flat+CSV (--loader fsc22):
      <root>/Audio Wise V1.0-*/<file.wav>
      <root>/Metadata-*/Metadata/*.csv
      Relative paths in the manifest use <ClassName>/<file.wav> format so
      the manifest is compatible with record_dataset.py and AudioFolderLoader.

The manifest (split_manifest.json) is written into the dataset root and should
be committed to git.  This freezes splits across sessions.

Usage
-----
    python tools/generate_split.py --input data/raw/fsc22_device
    python tools/generate_split.py --input data/raw/fsc22 --loader fsc22
    python tools/generate_split.py --input data/raw/fsc22 --loader fsc22 \\
        --seed 42 --train 0.70 --val 0.15 --test 0.15

Re-running is safe: the script warns if any previously-assigned file would move
to a different split, and exits without writing unless --force is given.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".aiff"}


def collect_files(root: Path) -> dict[str, list[str]]:
    """Return {class_name: [relative_path, ...]} for class-per-subfolder layout."""
    by_class: dict[str, list[str]] = defaultdict(list)
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith("."):
            continue
        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() in AUDIO_EXTS:
                by_class[class_dir.name].append(f"{class_dir.name}/{f.name}")
    return dict(by_class)


def collect_files_fsc22(root: Path) -> dict[str, list[str]]:
    """Return {class_name: [ClassName/filename, ...]} for the fsc22 flat+CSV layout.

    Relative paths use ClassName/filename.wav format so the manifest is
    compatible with record_dataset.py's resolve_candidates() and AudioFolderLoader.
    """
    import csv

    # Find the metadata CSV
    csv_files = list(root.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No metadata CSV found under {root}")
    csv_path = csv_files[0]

    # Find the flat audio directory
    audio_dirs = [
        p for p in root.rglob("*")
        if p.is_dir() and any(f.suffix.lower() == ".wav" for f in p.iterdir() if f.is_file())
    ]
    if not audio_dirs:
        raise FileNotFoundError(f"No audio directory found under {root}")
    audio_dir = audio_dirs[0]

    by_class: dict[str, list[str]] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            class_name   = row.get("Class Name", "").strip()
            dataset_fname = row.get("Dataset File Name", "").strip()
            if not class_name or not dataset_fname:
                continue
            path = audio_dir / dataset_fname
            if path.exists():
                by_class[class_name].append(f"{class_name}/{dataset_fname}")

    print(f"  CSV: {csv_path.name}")
    print(f"  Audio dir: {audio_dir.relative_to(root)}")
    return dict(by_class)


def stratified_split(
    by_class: dict[str, list[str]],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """Stratified split: each class is split proportionally."""
    rng = random.Random(seed)
    train, val, test = [], [], []
    for class_name in sorted(by_class):
        files = list(by_class[class_name])
        rng.shuffle(files)
        n = len(files)
        n_train = max(1, round(n * train_frac))
        n_val   = max(1, round(n * val_frac))
        # test gets the remainder to avoid rounding loss
        n_train = min(n_train, n - 2)   # guarantee at least 1 val and 1 test
        n_val   = min(n_val,   n - n_train - 1)
        train.extend(files[:n_train])
        val.extend(files[n_train:n_train + n_val])
        test.extend(files[n_train + n_val:])
    return train, val, test


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="python tools/generate_split.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input",  required=True,
                    help="Root of the dataset directory")
    ap.add_argument("--loader", default="audio_folder",
                    choices=["audio_folder", "fsc22"],
                    help="Dataset layout: audio_folder (class-per-subfolder, default) "
                         "or fsc22 (flat audio + metadata CSV)")
    ap.add_argument("--train",  type=float, default=0.70)
    ap.add_argument("--val",    type=float, default=0.15)
    ap.add_argument("--test",   type=float, default=0.15)
    ap.add_argument("--seed",   type=int,   default=42)
    ap.add_argument("--force",  action="store_true",
                    help="Overwrite existing manifest even if assignments change")
    args = ap.parse_args()

    root = Path(args.input)
    if not root.is_dir():
        print(f"ERROR: directory not found: {root}")
        raise SystemExit(1)

    frac_sum = args.train + args.val + args.test
    if abs(frac_sum - 1.0) > 1e-6:
        print(f"ERROR: train+val+test must sum to 1.0 (got {frac_sum:.3f})")
        raise SystemExit(1)

    if args.loader == "fsc22":
        by_class = collect_files_fsc22(root)
    else:
        by_class = collect_files(root)
    if not by_class:
        print("ERROR: no audio files found.")
        raise SystemExit(1)

    total = sum(len(v) for v in by_class.values())
    print(f"Found {total} files across {len(by_class)} classes:")
    for cls, files in sorted(by_class.items()):
        print(f"  {cls}: {len(files)} files")
    print()

    train, val, test = stratified_split(by_class, args.train, args.val, args.seed)
    print(f"Split (seed={args.seed}): train={len(train)}  val={len(val)}  test={len(test)}")

    manifest_path = root / "split_manifest.json"

    # Check for assignment changes vs existing manifest
    if manifest_path.exists() and not args.force:
        existing = json.loads(manifest_path.read_text())
        existing_assign = {}
        for split_name in ("train", "val", "test"):
            for f in existing.get(split_name, []):
                existing_assign[f] = split_name
        new_assign = {}
        for f in train: new_assign[f] = "train"
        for f in val:   new_assign[f] = "val"
        for f in test:  new_assign[f] = "test"

        changed = [f for f, s in existing_assign.items()
                   if f in new_assign and new_assign[f] != s]
        new_files = [f for f in new_assign if f not in existing_assign]

        if changed:
            print(f"\nWARNING: {len(changed)} existing file(s) would change split assignment:")
            for f in changed[:10]:
                print(f"  {f}: {existing_assign[f]} → {new_assign[f]}")
            if len(changed) > 10:
                print(f"  ... and {len(changed) - 10} more")
            print("\nRe-run with --force to overwrite, or use a different --seed.")
            raise SystemExit(1)

        if new_files:
            print(f"\n{len(new_files)} new file(s) added to manifest:")
            for f in new_files[:10]:
                print(f"  {f} → {new_assign[f]}")
            if len(new_files) > 10:
                print(f"  ... and {len(new_files) - 10} more")

    manifest = {
        "seed":  args.seed,
        "train": sorted(train),
        "val":   sorted(val),
        "test":  sorted(test),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {manifest_path}")
    print("Commit this file to git to freeze splits across sessions.")


if __name__ == "__main__":
    main()