#!/usr/bin/env python3
"""
Download and preprocess the PhysioNet EEG Motor Movement/Imagery dataset (EEGBCI)
into the `processed_data/` epochs format expected by the public training scripts.

This script intentionally does NOT commit the downloaded dataset to git.

Outputs (by default):
  data/public/processed_data/sub-XXX_run-R_processed-epo.fif

Usage:
  python scripts/download_public_dataset.py --subjects 1 2 3 --runs 3 7 11
  python scripts/download_public_dataset.py --subjects 1 5 10 --runs 6 10 14
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _parse_int_list(values: list[str]) -> list[int]:
    out: list[int] = []
    for v in values:
        v = v.strip()
        if not v:
            continue
        out.append(int(v))
    return out


def preprocess_subject_run(*, subject: int, run: int, raw_root: Path, out_dir: Path) -> Path:
    import mne
    from mne.datasets import eegbci

    # Download raw EDF(s)
    edf_paths = eegbci.load_data(subject, runs=[run], path=str(raw_root))
    if not edf_paths:
        raise RuntimeError(f"No EDF paths returned for subject={subject} run={run}")

    raw = mne.io.read_raw_edf(edf_paths[0], preload=True, verbose=False)
    eegbci.standardize(raw)  # channel naming conventions
    raw.set_montage("standard_1005", on_missing="ignore")

    # Events: EEGBCI uses Annotations like 'T0', 'T1', 'T2'.
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    if "T1" not in event_id or "T2" not in event_id:
        raise RuntimeError(f"Unexpected annotation set; got keys={sorted(event_id.keys())[:10]} ...")

    # Create epochs with the event labels expected by `data_utils.py`
    epochs = mne.Epochs(
        raw,
        events,
        event_id={"TASK1T1": event_id["T1"], "TASK1T2": event_id["T2"]},
        tmin=-1.0,
        tmax=4.0,
        baseline=(-1.0, 0.0),
        picks="eeg",
        preload=True,
        verbose=False,
        on_missing="ignore",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sub-{subject:03d}_run-{run}_processed-epo.fif"
    epochs.save(str(out_path), overwrite=True)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--subjects", nargs="+", default=["1"], help="Subject IDs to download (e.g. 1 2 3)")
    parser.add_argument("--runs", nargs="+", default=["3", "7", "11"], help="Run IDs (e.g. 3 7 11)")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2))),
        help="Parallel workers (downloads + preprocessing).",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/public/raw"),
        help="Where to download the raw EEGBCI files",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/public/processed_data"),
        help="Where to write processed epochs (.fif)",
    )
    args = parser.parse_args()

    subjects = _parse_int_list(args.subjects)
    runs = _parse_int_list(args.runs)

    tasks: list[tuple[int, int]] = [(s, r) for s in subjects for r in runs]

    # Parallelize end-to-end: each job downloads (if needed) and writes epochs.
    # This is typically faster than serial downloads, especially on multi-core machines.
    from joblib import Parallel, delayed

    def _one(s: int, r: int) -> str:
        out_path = preprocess_subject_run(subject=s, run=r, raw_root=args.raw_root, out_dir=args.processed_dir)
        return str(out_path)

    written = Parallel(n_jobs=int(args.n_jobs), prefer="processes")(delayed(_one)(s, r) for s, r in tasks)
    for p in written:
        print(f"[OK] Wrote {p}")

    print("\nDone.")
    print("To point training scripts at this directory, set:")
    print(f"  export PUBLIC_PROCESSED_DATA_DIR={args.processed_dir}")


if __name__ == "__main__":
    main()

