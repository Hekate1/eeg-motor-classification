#!/usr/bin/env python3
"""
Online Motor-Imagery BCI Pipeline
================================
A single-file reference implementation that turns *motor_imagery_experiment.py*
into a **closed-loop** experiment:

1. *Calibration stage* - collect `N_CAL` labelled trials, train CSP → SGD
   classifier on the fly.
2. *Feedback stage* - keep running the same cue loop but now display
   real-time classification feedback and **incrementally update** the model
   every `N_UPDATE` new trials.

Key Features
------------
• Works with **pre-trained weights** (`--weights csp.pkl clf.pkl`) but falls
  back to calibration if none provided.
• Uses **MNE + scikit-learn**: CSP for spatial filtering, `SGDClassifier`
  (`partial_fit`) for incremental updates.
• Asynchronous BrainFlow reader so GUI never blocks.
• Simple PsychoPy feedback: a green bar grows on the side corresponding to the
  predicted hand.
• Saves every run to `sub-XX_run-YY_online_raw.fif` plus updated weights.

Installation
------------
```bash
pip install brainflow psychopy mne scikit-learn joblib numpy
```

Usage
-----
```bash
python online_mi_pipeline.py --subject 01 --run 1 --trials 120 \
       --weights /path/to/csp.pkl /path/to/clf.pkl
```
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import List

import joblib
import mne
import numpy as np
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
from mne.decoding import CSP
from psychopy import core, event, visual
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from .gamified_feedback import GameScene
from .bar_feedback import BarFeedback
from .online_preprocessing import preprocess_epoch_data

###############################################################################
# ----------------------------- configuration --------------------------------
###############################################################################
N_CAL = 10000          # labelled trials used for initial calibration (even!)
N_UPDATE = 20       # re-fit model after every N_UPDATE new trials
EPOCH_TMIN = 0.0    # exclude cue onset artefacts (sec)
EPOCH_TMAX = 2.0    # imagination window (sec)
BANDPASS = (8, 30)  # mu-beta band for MI
###############################################################################


def make_trial_sequence(n_trials: int, seed: int) -> List[int]:
    if n_trials % 2:
        raise ValueError("Number of trials must be even (balance L/R).")
    trials = [1]*(n_trials//2) + [2]*(n_trials//2)
    rng = random.Random(seed); rng.shuffle(trials)
    return trials


def connect_board(port: str):
    params = BrainFlowInputParams(); params.serial_port = port
    board = BoardShim(BoardIds.CYTON_DAISY_BOARD, params)
    board.prepare_session(); board.start_stream()
    return board


def epoch_data(raw: mne.io.Raw, events, sfreq):
    """Return X (n_epochs, n_ch, n_times) and y labels."""
    epochs = mne.Epochs(raw, events, event_id={"Left":1,"Right":2},
                        tmin=EPOCH_TMIN, tmax=EPOCH_TMAX, baseline=None,
                        proj=False, picks="eeg", preload=True,
                        on_missing='ignore')
    X = epochs.get_data(); y = epochs.events[:,2]
    return X, y


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--port", help="/dev/cu.usbserial-XYZ (auto if omitted)")
    parser.add_argument("--weights", nargs="*", help="paths to csp.pkl clf.pkl scaler.pkl")
    parser.add_argument("--model", help="Path to pretrained HybridModelClassifier .pt file for offline inference")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output files")
    parser.add_argument("--game", action="store_true", help="Use gamified feedback game instead of bars")
    args = parser.parse_args()

    # Determine if using offline pretrained hybrid model
    use_offline_model = args.model is not None
    if use_offline_model:
        try:
            # Optional dependency: only needed when you pass --model
            from pipeline.public.hybrid_cnn_transformer import HybridModelClassifier  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "Failed to import the ML pipeline package needed for --model. "
                "Install the repo package (e.g. `pip install -e .`) and ML deps."
            ) from exc
        classifier, metadata = HybridModelClassifier.load_model(args.model)
        print(f"[INFO] Loaded pretrained HybridModelClassifier from {args.model}")
        # No incremental calibration or update, will use classifier.predict on each epoch

    use_game = args.game

    # Prevent accidental overwrite of existing output file
    out_fname = f"sub-{args.subject}_run-{args.run:02d}_online_raw.fif"
    if Path(out_fname).exists() and not args.overwrite:
        sys.exit(f"[ERR] Output file {out_fname} already exists. Use --overwrite to overwrite.")

    # -------- trial list ----------
    trials = make_trial_sequence(args.trials, seed=args.run)

    # -------- connect OpenBCI board ----------
    port = args.port or next(Path("/dev").glob("cu.usbserial-*"), None)
    if port is None:
        sys.exit("[ERR] Could not auto-detect OpenBCI serial port; use --port.")
    board = connect_board(str(port))
    # -------- verify streaming connectivity ----------
    print("[INFO] Verifying board streaming...")
    time.sleep(1.0)
    count = board.get_board_data_count()
    if count == 0:
        board.stop_stream(); board.release_session()
        sys.exit("[ERR] No data received from board. Please ensure headset is properly connected to the dongle.")
    else:
        print("[INFO] Board streaming OK.")

    # -------- PsychoPy setup ----------
    win = visual.Window((800,600), fullscr=False, color="white", units="pix")
    if use_game:
        scene = GameScene(win)
        fixation = visual.TextStim(win, text='+', height=80, color='black')
    else:
        # Use shared BarFeedback for non-game feedback
        bar_feedback = BarFeedback(win)
        fixation = visual.TextStim(win, text='+', height=80, color='black')
        cue = visual.TextStim(win, height=120, color='black')
        scene = None

    # -------- initialise model ----------
    sfreq = BoardShim.get_sampling_rate(BoardIds.CYTON_DAISY_BOARD.value)
    descr = BoardShim.get_board_descr(BoardIds.CYTON_DAISY_BOARD.value)
    eeg_rows = descr['eeg_channels']

    csp  = CSP(n_components=6, reg=None, log=True, norm_trace=False)
    clf  = SGDClassifier(loss="log_loss", max_iter=1000, learning_rate="optimal")
    scaler = StandardScaler()
    # buffer for all training features and labels
    X_train_all = None
    y_train_all = None
    # load pre-trained CSP, classifier, and scaler if provided
    have_weights = args.weights and len(args.weights)==3 and Path(args.weights[0]).exists() and Path(args.weights[2]).exists()
    if have_weights:
        csp = joblib.load(args.weights[0])
        clf = joblib.load(args.weights[1])
        scaler = joblib.load(args.weights[2])
        print("[INFO] Pre-trained CSP, classifier, and scaler loaded → skipping calibration.")
    else:
        # Attempt to load previous runs for calibration
        prev_files = sorted(Path('.').glob(f"sub-{args.subject}_run-*_online_raw.fif"))
        prev_runs = []
        for f in prev_files:
            print(f)
            stem = f.stem  # e.g. sub-01_run-02_online_raw
            try:
                run_part = stem.split('_')[1]  # 'run-02'
                run_num = int(run_part.split('-')[1])
            except (IndexError, ValueError):
                continue
            if run_num < args.run:
                prev_runs.append(f)
        if prev_runs:
            print(f"[INFO] Loading calibration data from previous runs: {prev_runs}")
            X_list, y_list = [], []
            for f in prev_runs:
                raw_prev = mne.io.read_raw_fif(str(f), preload=True, verbose=False)
                events = mne.find_events(raw_prev, stim_channel='STI 014', shortest_event=1)
                X_prev, y_prev = epoch_data(raw_prev, events, sfreq)
                X_list.append(X_prev.reshape(X_prev.shape[0], X_prev.shape[1], -1))
                y_list.append(y_prev)
            X_all = np.vstack(X_list)
            y_all = np.concatenate(y_list)
            X_csp = csp.fit_transform(X_all, y_all)
            X_scaled = scaler.fit_transform(X_csp)
            clf.fit(X_scaled, y_all)
            # store training data for future updates
            X_train_all = X_csp
            y_train_all = y_all
            have_weights = True
            print("[OK] Calibration from previous runs finished → switching to feedback mode.")
        else:
            print(f"[INFO] No previous runs found → collecting {N_CAL} calibration trials first.")

    # wait for Enter to begin experiment and show initial fixation
    print("[INFO] Press Enter to begin experiment.")
    if use_game:
        instr = "Game: Perform the task with the hand on the side with the portal highlighted in green\n\nPress Enter to begin experiment"
    else:
        instr = "Press Enter to begin experiment"
    visual.TextStim(win, text=instr, height=40, color="black").draw()
    win.flip()
    event.waitKeys(keyList=["return"])
    if use_game:
        scene.baseline(2.0)
    else:
        fixation.draw(); win.flip(); core.wait(2.0)

    # storage for incremental retraining
    X_buf, y_buf = [], []
    # storage for performance metrics
    true_labels = []
    pred_labels = []
    true_probs = []
    # -------- main experiment loop ----------
    raw_buffer = None
    # Flag to print adjustment only once
    printed_adjustment = False
    cue_clock = core.Clock()
    for i, code in enumerate(trials,1):
        # present cue
        if use_game:
            scene.get_ready(code)
            cue_clock.reset()
        else:
            label = "LEFT" if code==1 else "RIGHT"
            cue.text = label; cue.draw(); win.flip(); cue_clock.reset()
        board.insert_marker(code)

        # let the imagination window elapse (EPOCH_TMAX seconds)
        core.wait(EPOCH_TMAX + 0.5)

        # pull data so far
        # fetch only new data samples and clear buffer to avoid duplicates
        count = board.get_board_data_count()
        data = board.get_board_data(count)
        if raw_buffer is None:
            # first loop: create Raw buffer
            ch_names=[f"C{k+1}" for k in range(len(eeg_rows))]+['STI 014']
            info=mne.create_info(ch_names, sfreq, ch_types=['eeg']*16+['stim'])
            raw_buffer = mne.io.RawArray(np.zeros((17,1)), info, verbose=False)
        # append new block
        eeg = data[eeg_rows]*1e-6; stim=data[descr['marker_channel']][None,:]
        chunk = np.vstack([eeg, stim])

        raw_buffer.append(mne.io.RawArray(chunk, raw_buffer.info, verbose=False))

        # after each trial, if we have markers >= 2 --> try epoching last trial
        events = mne.find_events(raw_buffer, stim_channel='STI 014', shortest_event=1)
        if len(events) >= i:  # we have at least i events in buffer
            try:
                X_epoch, y_epoch = preprocess_epoch_data(raw_buffer, events[-1:, :], EPOCH_TMIN, EPOCH_TMAX)
                X_buf.append(X_epoch.squeeze()); y_buf.append(y_epoch.item())
                print(f"Trial {i}, type {y_epoch.item()}")
            except Exception as e:
                print(f"[WARN] Skipping epoch {i} due to error: {e}\n-------")
                fixation.draw(); win.flip()
                core.wait(1.0)  # rest
                continue

        try:
            # -------- training / feedback logic ----------
            if use_offline_model and X_epoch is not None:
                # Ensure epoch matches model's expected time dimension
                n_times_model = classifier.model.n_times
                t_cur = X_epoch.shape[2]
                if not printed_adjustment and t_cur != n_times_model:
                    print(f"[INFO] Adjusting epoch time dimension from {t_cur} to {n_times_model}")
                    printed_adjustment = True
                if t_cur > n_times_model:
                    X_epoch = X_epoch[:, :, :n_times_model]
                elif t_cur < n_times_model:
                    pad_width = ((0,0), (0,0), (0, n_times_model - t_cur))
                    X_epoch = np.pad(X_epoch, pad_width, mode='constant', constant_values=0)
                # Provide dummy subject index for CSP per-subject transform
                subject_indices = np.zeros(X_epoch.shape[0], dtype=int)
                res = classifier.evaluate(X_epoch, np.array([y_epoch.item()]), subject_indices=subject_indices)
                proba = res['probabilities'][0]
                print(f"Proba: {proba}")
                if use_game:
                    scene.update(code, np.argmax(proba) + 1, proba)
                else:
                    bar_feedback.update(proba)
                    bar_feedback.draw()
                    fixation.draw(); win.flip()
                core.wait(1.0)
                if use_game:
                    scene.baseline(1.0)
                else:
                    fixation.draw(); win.flip(); core.wait(1.0)
                true_labels.append(y_epoch.item())
                pred_label = np.argmax(proba) + 1
                pred_labels.append(pred_label)
                true_probs.append(proba[y_epoch.item()-1])
                continue
            
            if not have_weights and len(X_buf)>=N_CAL:
                # initial calibration finished
                X=np.array(X_buf); y=np.array(y_buf)
                raw_feat = X.reshape(X.shape[0], X.shape[1], -1)
                X_csp = csp.fit_transform(raw_feat, y)
                X_scaled = scaler.fit_transform(X_csp)
                clf.fit(X_scaled, y)
                # store training data for future updates
                X_train_all = X_csp
                y_train_all = y
                have_weights=True
                X_buf.clear(); y_buf.clear()
                print("[OK] Initial model trained → switching to feedback mode.")

            elif have_weights and len(X_buf)>=N_UPDATE:
                # periodic update
                X_new = np.array(X_buf); y_new = np.array(y_buf)
                feats_new = csp.transform(X_new.reshape(X_new.shape[0], X_new.shape[1], -1))
                # accumulate across all data
                if X_train_all is None:
                    X_train_all = feats_new
                    y_train_all = y_new
                else:
                    X_train_all = np.vstack([X_train_all, feats_new])
                    y_train_all = np.concatenate([y_train_all, y_new])
                # re-fit scaler and classifier on entire dataset
                X_scaled_all = scaler.fit_transform(X_train_all)
                clf.fit(X_scaled_all, y_train_all)
                # clear buffers
                X_buf.clear(); y_buf.clear()
                print(f"[INFO] Model re-trained on all data (n_trials={len(y_train_all)})")

            # --- real-time feedback (if model ready) ---
            if have_weights and X_epoch is not None:
                feat = csp.transform(X_epoch.reshape(1,16,-1))
                feat_scaled = scaler.transform(feat)
                proba = clf.predict_proba(feat_scaled)[0]  # [p_left, p_right]
                print(f"Proba: {proba}")
                if use_game:
                    scene.update(code, np.argmax(proba) + 1, proba)
                else:
                    bar_feedback.update(proba)
                    bar_feedback.draw()
                    fixation.draw(); win.flip()
                core.wait(1.0)  # show feedback one second
                if use_game:
                    scene.baseline(1.0)
                else:
                    fixation.draw(); win.flip(); core.wait(1.0)
                # record performance metrics
                true_labels.append(y_epoch.item())
                pred_label = np.argmax(proba) + 1
                pred_labels.append(pred_label)
                true_probs.append(proba[y_epoch.item()-1])
            else:
                if use_game:
                    scene.baseline(1.0)
                else:
                    fixation.draw(); win.flip(); core.wait(1.0)  # rest

        except Exception as e:
            print(f"[ERROR] {e}")
            if raw_buffer is not None:
                err_fname = f"data/sub-{args.subject}_run-{args.run:02d}_error_raw.fif"
                raw_buffer.save(err_fname, overwrite=True)
                print(f"[INFO] Partial data saved to {err_fname}")
            sys.exit(1)

        if event.getKeys(["escape"]):
            print("[ABORT] Escape pressed. Ending early.")
            break

    # -------- performance summary --------
    if true_labels:
        acc = sum(1 for t,p in zip(true_labels, pred_labels) if t==p) / len(true_labels)
        avg_prob = sum(true_probs) / len(true_probs)
        print(f"[RESULT] Raw accuracy: {acc*100:.2f}%")
        print(f"[RESULT] Average probability for true class: {avg_prob:.3f}")
    else:
        print("[RESULT] No predictions were made.")

    # -------- save raw + weights ----------
    try:
        fname_raw = f"data/sub-{args.subject}_run-{args.run:02d}_online_raw.fif"
        raw_buffer.save(fname_raw, overwrite=True)
        if not use_offline_model:
            joblib.dump(csp, f"data/sub-{args.subject}_csp.pkl")
            joblib.dump(clf, f"data/sub-{args.subject}_clf.pkl")
            joblib.dump(scaler, f"data/sub-{args.subject}_scaler.pkl")
            print(f"[DONE] Data + weights + scaler saved to {fname_raw}, data/sub-{args.subject}_{{csp,clf,scaler}}.pkl")
        else:
            print(f"[DONE] Data saved to {fname_raw}")
    except Exception as e:
        print(f"[ERROR] Error saving data: {e}")
        if raw_buffer is not None:
            err_fname = f"data/sub-{args.subject}_run-{args.run:02d}_error_raw.fif"
            raw_buffer.save(err_fname, overwrite=True)
            print(f"[INFO] Partial data saved to {err_fname}")
            sys.exit(1)

    board.stop_stream(); board.release_session(); win.close(); core.quit()


if __name__ == "__main__":
    main()
