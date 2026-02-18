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
import time
import re
from typing import Iterable, Union, Any


def _parse_int_list(values: list[str]) -> list[int]:
    out: list[int] = []
    for v in values:
        v = v.strip()
        if not v:
            continue
        out.append(int(v))
    return out


_RUN_RE = re.compile(r"R(\d+)\.edf$", re.IGNORECASE)


def _infer_run_from_path(path: Union[str, os.PathLike[str], Any]) -> int:
    path_s = str(path)
    m = _RUN_RE.search(path_s)
    if not m:
        raise ValueError(f"Could not infer run number from EDF filename: {path_s}")
    return int(m.group(1))


def _flatten_paths(paths: Iterable[Any]) -> list[Any]:
    out: list[Any] = []
    for p in paths:
        if isinstance(p, (list, tuple)):
            out.extend(list(p))
        else:
            out.append(p)
    return out


def preprocess_subject(
    *,
    subject: int,
    runs: list[int],
    raw_root: Path,
    out_dir: Path,
    update_mne_config: bool,
) -> list[Path]:
    import mne
    from mne.datasets import eegbci

    t0 = time.perf_counter()

    # Download raw EDF(s) (MNE caches under raw_root/MNE-eegbci-data/...).
    #
    # Important: when running in parallel, we must avoid `input()` prompts.
    # Setting update_path=False prevents MNE from prompting to update its global config.
    try:
        edf_paths = eegbci.load_data(
            subject,
            runs=runs,
            path=str(raw_root),
            update_path=True if update_mne_config else False,
        )
    except TypeError:
        # Backward compatibility with older MNE versions (no update_path arg).
        edf_paths = eegbci.load_data(subject, runs=runs, path=str(raw_root))
    if not edf_paths:
        raise RuntimeError(f"No EDF paths returned for subject={subject} runs={runs}")
    edf_paths = _flatten_paths(edf_paths)

    t_download = time.perf_counter() - t0
    t1 = time.perf_counter()

    written: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    for edf_path in edf_paths:
        run = _infer_run_from_path(edf_path)
        if run not in runs:
            continue

        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
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

        out_path = out_dir / f"sub-{subject:03d}_run-{run}_processed-epo.fif"
        epochs.save(str(out_path), overwrite=True)
        written.append(out_path)

    t_process = time.perf_counter() - t1
    print(f"[SUBJ {subject:03d}] downloaded {len(edf_paths)} EDF(s) in {t_download:.1f}s; processed {len(written)} run(s) in {t_process:.1f}s")
    return written

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
    parser.add_argument(
        "--update-mne-config",
        action="store_true",
        help="Allow MNE to update its global EEGBCI dataset path config (avoids prompt).",
    )
    args = parser.parse_args()

    subjects = _parse_int_list(args.subjects)
    runs = _parse_int_list(args.runs)

    tasks: list[int] = list(subjects)

    # Parallelize by subject: each worker downloads all requested runs for a
    # subject in a single MNE call, then preprocesses/saves per run.
    from joblib import Parallel, delayed

    def _one(s: int) -> list[str]:
        written = preprocess_subject(
            subject=s,
            runs=runs,
            raw_root=args.raw_root,
            out_dir=args.processed_dir,
            update_mne_config=bool(args.update_mne_config),
        )
        return [str(p) for p in written]

    written_lists = Parallel(n_jobs=int(args.n_jobs), prefer="processes")(delayed(_one)(s) for s in tasks)
    for written in written_lists:
        for p in written:
            print(f"[OK] Wrote {p}")

    print("\nDone.")
    print("To point training scripts at this directory, set:")
    print(f"  export PUBLIC_PROCESSED_DATA_DIR={args.processed_dir}")


if __name__ == "__main__":
    main()

