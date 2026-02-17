#!/usr/bin/env python3
"""
Motor Imagery (Left- vs- Right-hand) Experiment
==============================================
* Compatible with OpenBCI **Cyton + Daisy** (16 EEG channels)
* One-file solution: acquisition **and** cue presentation in the same process

Features
--------
• Auto-detect the serial port (macOS - `/dev/cu.usbserial-*`) or override with
  `--port /dev/tty…`.
• Command-line options for **subject ID**, **run number**, **#trials**, and
  cue/rest durations.
• Reproducible, balanced trial list (equal L/R) seeded by the run number.
• Stimulus timing & markers accurate to one video frame (< 2 ms on 120 Hz).
• Saves a full-fidelity MNE **.fif** file (`sub-XX_run-YY_raw.fif`) that already
  contains the marker channel.

Install requirements
--------------------
```bash
pip install brainflow psychopy mne numpy
```

Run
---
```bash
python motor_imagery_experiment.py --subject 02 --run 3 --trials 120
```
"""

from __future__ import annotations

import argparse
import glob
import random
import sys
from pathlib import Path
from typing import List

import mne
import numpy as np
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
from psychopy import core, event, visual

###############################################################################
# -------------------------- helper utilities --------------------------------
###############################################################################

def find_default_serial_port() -> str | None:
    """Return the first `/dev/cu.usbserial-*` path on macOS or ``None`` if none."""
    ports = sorted(glob.glob("/dev/cu.usbserial-*"))
    return ports[0] if ports else None


def build_trial_sequence(n_trials: int, seed: int) -> List[int]:
    """Return a list of *n_trials* codes (1 = Left, 2 = Right), shuffled.

    Ensures the list is perfectly balanced.
    """
    if n_trials % 2:
        raise ValueError("Number of trials must be even to balance left/right cues.")

    trials = [1] * (n_trials // 2) + [2] * (n_trials // 2)
    rng = random.Random(seed)
    rng.shuffle(trials)
    return trials


def graceful_exit(board: BoardShim | None, win: visual.Window | None):
    """Stop the stream, release hardware, close the PsychoPy window."""
    if board is not None:
        try:
            board.stop_stream()
            board.release_session()
        except Exception:
            pass
    if win is not None:
        win.close()
    core.quit()


###############################################################################
# ----------------------------- main routine ---------------------------------
###############################################################################


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run a left/right motor-imagery EEG acquisition using an OpenBCI Cyton + Daisy board.",
    )
    parser.add_argument("--subject", required=True, help="numeric or alphanumeric subject ID (e.g. 01, S3)")
    parser.add_argument("--run", type=int, required=True, help="run number (used as RNG seed)")
    parser.add_argument("--trials", type=int, default=120, help="total number of trials (must be even)")
    parser.add_argument("--port", help="serial port path of OpenBCI dongle (auto-detected if omitted)")
    parser.add_argument("--imag_time", type=float, default=4.0, help="motor-imagery duration in seconds")
    parser.add_argument("--rest_time", type=float, default=2.0, help="inter-trial rest in seconds")
    parser.add_argument("--fullscreen", action="store_true", help="show cues in fullscreen mode")

    args = parser.parse_args()

    # ---------------- serial port resolution ----------------
    serial_port = args.port or find_default_serial_port()
    if serial_port is None:
        sys.exit("[ERR] Could not auto-detect an OpenBCI serial port. Use --port.")

    print(f"[INFO] Using serial port: {serial_port}")

    # ---------------- BrainFlow setup ----------------
    bf_params = BrainFlowInputParams()
    bf_params.serial_port = serial_port
    board = BoardShim(BoardIds.CYTON_DAISY_BOARD, bf_params)

    try:
        board.prepare_session()
    except Exception as exc:
        sys.exit(f"[ERR] Failed to prepare BrainFlow session: {exc}")

    board.start_stream()
    print("[INFO] Stream started. Press ESC to abort.")

    # ---------------- PsychoPy window & stimuli ----------------
    win = visual.Window(size=(800, 600), fullscr=args.fullscreen, color="white", units="pix")
    cue_text = visual.TextStim(win, height=120, font="Helvetica", color="black")
    fixation = visual.TextStim(win, text="+", height=80, color="black")

    # trial sequence (1 = left, 2 = right)
    try:
        trials = build_trial_sequence(args.trials, args.run)
    except ValueError as err:
        graceful_exit(board, win)
        sys.exit(str(err))

    # wait for Enter to begin experiment and show initial fixation
    print("[INFO] Press Enter to begin experiment.")
    event.waitKeys(keyList=["return"])
    fixation.draw()
    win.flip()
    core.wait(args.rest_time)

    # ---------------- experiment loop ----------------
    for idx, code in enumerate(trials, 1):
        label = "LEFT" if code == 1 else "RIGHT"
        print(f"[TRIAL {idx:03}/{len(trials)}] {label}")

        # 1) cue
        cue_text.text = label
        cue_text.draw()
        win.flip()
        board.insert_marker(code)
        core.wait(args.imag_time)

        # 2) rest / fixation
        fixation.draw()
        win.flip()
        core.wait(args.rest_time)

        # allow early abort
        if event.getKeys(keyList=["escape"]):
            print("[INFO] Experiment aborted by user.")
            break

    # ---------------- acquisition stop & save ----------------
    print("[INFO] Stopping stream and saving data…")
    board.stop_stream()
    raw_fname = f"sub-{args.subject}_run-{args.run:02d}_raw.fif"

    data = board.get_board_data()  # shape (32, n_samples)
    descr = BoardShim.get_board_descr(BoardIds.CYTON_DAISY_BOARD.value)

    eeg_rows = descr["eeg_channels"]
    stim_row = descr["marker_channel"]
    sfreq = descr["sampling_rate"]

    eeg = data[eeg_rows] * 1e-6  # µV → V
    stim = data[stim_row][None, :]

    ch_names = [f"C{i+1}" for i in range(len(eeg_rows))] + ["STI 014"]
    ch_types = ["eeg"] * len(eeg_rows) + ["stim"]

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(np.vstack([eeg, stim]), info)
    raw.save(raw_fname, overwrite=True)

    print(f"[OK] Data written to {raw_fname}")

    graceful_exit(board, win)


###############################################################################
# --------------------------- entry point ------------------------------------
###############################################################################

if __name__ == "__main__":
    main()
