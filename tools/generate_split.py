"""
tools/generate_split.py — generate a reproducible train/val/test split manifest
for a class-per-subfolder device recording directory.

The manifest (split_manifest.json) is written into the dataset root and should
be committed to git.  This freezes splits across sessions: existing file
assignments never change when new recordings are added.

Usage
-----
    python tools/generate_split.py --input data/raw/fsc22_device
    python tools/generate_split.py --input data/raw/fsc22_device --train 0.70 --val 0.15 --test 0.15

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
    """Return {class_name: [relative_path, ...]} sorted deterministically."""
    by_class: dict[str, list[str]] = defaultdict(list)
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith("."):
            continue
        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() in AUDIO_EXTS:
                by_class[class_dir.name].append(f"{class_dir.name}/{f.name}")
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
                    help="Root of the device recording directory")
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