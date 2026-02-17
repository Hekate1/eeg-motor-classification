#!/usr/bin/env python3
"""
Sanitize MNE FIF files for public sharing.

What it does (best-effort):
- Removes measurement date (meas_date)
- Clears subject and experimenter fields
- Clears device_info (can include serial-like identifiers)

Usage:
  python scripts/sanitize_fif_metadata.py data/self --overwrite
  python scripts/sanitize_fif_metadata.py data/self/sub-01_run-01_online_raw.fif --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


def iter_fif_paths(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    yield from sorted(path.rglob("*.fif"))


def sanitize_one(path: Path, *, dry_run: bool, overwrite: bool) -> None:
    import mne

    raw = mne.io.read_raw_fif(str(path), preload=False, verbose=False)

    # Best-effort anonymization. Prefer MNE helper when available.
    try:
        raw.anonymize(daysback=365 * 200, keep_his=False)
    except Exception:
        # Fallback for older MNE or unexpected structures.
        try:
            raw.set_meas_date(None)
        except Exception:
            pass

        info = raw.info
        for k in ("subject_info", "experimenter", "proj_name", "description", "device_info"):
            try:
                info[k] = None
            except Exception:
                pass

    if dry_run:
        print(f"[DRY-RUN] Would overwrite sanitized FIF: {path}")
        return

    if not overwrite:
        raise SystemExit(f"Refusing to overwrite in-place without --overwrite: {path}")

    # MNE intentionally forbids saving to the same file path. Write to a temp
    # sibling and then atomically replace.
    # Avoid MNE naming-convention warnings by keeping a `*_raw*.fif` ending.
    if path.name.endswith("_raw.fif"):
        tmp_path = path.with_name(path.name.replace("_raw.fif", "_raw_tmp.fif"))
    else:
        tmp_path = path.with_suffix(".tmp_raw.fif")
    if tmp_path.exists():
        tmp_path.unlink()

    raw.save(str(tmp_path), overwrite=True, verbose=False)
    tmp_path.replace(path)
    print(f"[OK] Sanitized FIF (in-place): {path}")


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("path", type=Path, help="FIF file or a directory containing FIFs")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without writing files")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting in-place")
    args = parser.parse_args()

    paths = list(iter_fif_paths(args.path))
    if not paths:
        raise SystemExit(f"No .fif files found under {args.path}")

    for p in paths:
        sanitize_one(p, dry_run=args.dry_run, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

