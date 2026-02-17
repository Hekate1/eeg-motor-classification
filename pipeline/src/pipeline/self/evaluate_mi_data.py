#!/usr/bin/env python3
"""
Evaluate & Visualise Motor-Imagery FIF Files
===========================================
Given one or more **Raw FIF** files created by *motor_imagery_experiment.py*,
this script provides quick sanity-checks:

1. Counts of LEFT vs. RIGHT events per run.
2. Interactive raw browser (for manual inspection).
3. Power-spectral density (PSD) for spotting noise / line hum.
4. Event timeline plot - do markers line up evenly?
5. (Optional) Simple sensor-level ERPs (--erp flag).

Run Examples
------------
```bash
# look at a single run
python evaluate_mi_data.py sub-02_run-03_raw.fif

# batch over a folder, suppressing interactive plots
python evaluate_mi_data.py data/sub-02_*.fif --no-browser
```
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mne
import numpy as np
from matplotlib import pyplot as plt

###############################################################################
# ----------------------------- helpers --------------------------------------
###############################################################################

def count_events(raw: mne.io.Raw) -> dict[int, int]:
    """Return a dict mapping marker code → count."""
    events = mne.find_events(raw, stim_channel="STI 014", shortest_event=1)
    return {code: int(np.sum(events[:, 2] == code)) for code in np.unique(events[:, 2])}


def plot_summary(raw: mne.io.Raw, browser: bool = True, erp: bool = False):
    """Interactive plots for a single Raw object."""
    print(" → Raw duration: {:.1f} s".format(raw.times[-1]))

    if browser:
        raw.plot(title="Raw Browser", duration=10, n_channels=20, scalings={"eeg": 5e-5})
        raw.compute_psd(fmax=60).plot()
        plt.title("PSD")

    # Events timeline
    events = mne.find_events(raw, stim_channel="STI 014", shortest_event=1)
    mne.viz.plot_events(events, sfreq=raw.info["sfreq"])

    # ERP / time-locked average (optional)
    if erp:
        epochs = mne.Epochs(
            raw,
            events,
            event_id={"Left": 1, "Right": 2},
            tmin=-0.2,
            tmax=1.0,
            baseline=None,
            preload=True,
        )
        evoked_left = epochs["Left"].average()
        evoked_right = epochs["Right"].average()
        mne.viz.plot_compare_evokeds(
            {"Left": evoked_left, "Right": evoked_right},
            picks="eeg",
            ci=False,
            combine=None,
            title="Sensor-level ERPs (mean over EEG channels)",
        )
        # Additional ERP topomap plots
        fig = evoked_left.plot_topomap(times=np.linspace(0.1, 0.6, 5), ch_type="eeg")
        fig.suptitle("Left ERP Topomap")
        fig = evoked_right.plot_topomap(times=np.linspace(0.1, 0.6, 5), ch_type="eeg")
        fig.suptitle("Right ERP Topomap")

    plt.show(block=True)

###############################################################################
# ----------------------------- main -----------------------------------------
###############################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Quick evaluation of motor-imagery runs saved as FIF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("files", nargs="+", help="FIF files to inspect (supports glob patterns)")
    parser.add_argument("--no-browser", action="store_true", help="skip the interactive raw browser & PSD plots")
    parser.add_argument("--erp", action="store_true", help="compute and plot simple ERPs (Left vs Right)")

    args = parser.parse_args()

    # Resolve globs → list of distinct files
    fif_paths = {str(p) for pattern in args.files for p in Path().glob(pattern)}
    if not fif_paths:
        parser.error("No FIF files matched.")

    for fname in sorted(fif_paths):
        print(f"\n=== {fname} ===")
        raw = mne.io.read_raw_fif(fname, preload=True, verbose="ERROR")

        # Set channel names and standard 10-20 montage
        ch_names = ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2', 'F7', 'F8', 'F3', 'F4', 'T7', 'T8', 'P3', 'P4']

        raw.rename_channels(dict(zip(raw.ch_names, ch_names)))
        raw.set_montage(mne.channels.make_standard_montage('standard_1020'))

        print(" → Applying filters: 60 Hz notch, 1-60 Hz bandpass")
        raw.notch_filter(60)
        raw.filter(1., 60., fir_design='firwin')

        counts = count_events(raw)
        print("Event counts:")
        for code in (1, 2):
            label = "Left" if code == 1 else "Right"
            print(f"  {label:>5}: {counts.get(code, 0)}")

        plot_summary(raw, browser=not args.no_browser, erp=args.erp)

if __name__ == "__main__":
    main()
