#!/usr/bin/env python3
"""
Demo playback of the Online MI Pipeline using a pre-recorded .fif file.
Loads a pretrained HybridModelClassifier, epochs trials, and displays real-time feedback predictions.
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import mne
from psychopy import core, event, visual

# Import epoching and feedback scene from experiment
exp_src = Path(__file__).resolve().parent
sys.path.insert(0, str(exp_src))
from online_mi_pipeline import epoch_data, EPOCH_TMIN, EPOCH_TMAX
from online_preprocessing import preprocess_epoch_data
from gamified_feedback import GameScene
from bar_feedback import BarFeedback

# Import model-side classifier
model_src = exp_src.parent / 'model' / 'src'
sys.path.insert(0, str(model_src))
from hybrid_cnn_transformer import HybridModelClassifier


def main():
    parser = argparse.ArgumentParser(description='Demo playback for Online MI Pipeline')
    parser.add_argument('--fif', type=Path, required=True, help='Path to raw .fif file')
    parser.add_argument('--model', required=True, help='Path to pretrained HybridModelClassifier .pt file')
    parser.add_argument('--game', action='store_true', help='Use gamified feedback instead of bars')
    parser.add_argument('--delay', type=float, default=2.0, help='Feedback display duration (seconds)')
    parser.add_argument('--stats-only', action='store_true', help='Calculate accuracy stats only without UI')
    args = parser.parse_args()

    # Load pretrained model
    classifier, metadata = HybridModelClassifier.load_model(args.model)
    print(f"[INFO] Loaded model from {args.model}")
    if args.stats_only:
        # Calculate accuracy stats without any UI
        if not args.fif.exists():
            print(f"File not found: {args.fif}")
            sys.exit(1)
        raw = mne.io.read_raw_fif(str(args.fif), preload=True, verbose=False)
        sfreq = raw.info['sfreq']
        events = mne.find_events(raw, stim_channel='STI 014', shortest_event=1)
        X_all, y_all = preprocess_epoch_data(raw, events, EPOCH_TMIN, EPOCH_TMAX)
        # Ensure batch matches model's expected time dimension
        n_times_model = classifier.model.n_times
        t_current = X_all.shape[2]
        if t_current != n_times_model:
            print(f"[INFO] Adjusting time dimension of all trials from {t_current} to {n_times_model}")
        if t_current > n_times_model:
            X_all = X_all[:, :, :n_times_model]
        elif t_current < n_times_model:
            pad_width = ((0,0), (0,0), (0, n_times_model - t_current))
            X_all = np.pad(X_all, pad_width, mode='constant', constant_values=0)
        subject_indices = np.zeros(len(y_all), dtype=int)
        res = classifier.evaluate(X_all, y_all, subject_indices=subject_indices)
        proba = res['probabilities']
        pred_labels = np.argmax(proba, axis=1) + 1
        accuracy = np.mean([t == p for t, p in zip(y_all, pred_labels)])
        print(f"[STATS] Accuracy: {accuracy*100:.2f}%")
        # Compute prediction confidence statistics
        confidences = np.max(proba, axis=1)
        correct_mask = (y_all == pred_labels)
        # Stats for correct predictions
        if correct_mask.any():
            corr_conf = confidences[correct_mask]
            print(f"[STATS] Correct predictions: {correct_mask.sum()}/{len(y_all)}")
            print(f"[STATS] Confidence (correct) — mean: {corr_conf.mean()*100:.2f}%, std: {corr_conf.std()*100:.2f}%, min: {corr_conf.min()*100:.2f}%, max: {corr_conf.max()*100:.2f}%")
        else:
            print("[STATS] No correct predictions to compute confidence stats")
        # Stats for incorrect predictions
        if (~correct_mask).any():
            inc_conf = confidences[~correct_mask]
            print(f"[STATS] Incorrect predictions: {(~correct_mask).sum()}/{len(y_all)}")
            print(f"[STATS] Confidence (incorrect) — mean: {inc_conf.mean()*100:.2f}%, std: {inc_conf.std()*100:.2f}%, min: {inc_conf.min()*100:.2f}%, max: {inc_conf.max()*100:.2f}%")
        else:
            print("[STATS] No incorrect predictions to compute confidence stats")
        sys.exit(0)

    # Setup PsychoPy window
    win = visual.Window((800,600), fullscr=False, color='white', units='pix')
    if args.game:
        scene = GameScene(win)
        fixation = visual.TextStim(win, text='+', height=80, color='black')
        cue = None
    else:
        # Use shared BarFeedback for non-game feedback
        bar_feedback = BarFeedback(win)
        fixation = visual.TextStim(win, text='+', height=80, color='black')
        cue = visual.TextStim(win, text='', height=120, color='black')
        scene = None

    # Load and epoch .fif data
    if not args.fif.exists():
        print(f"File not found: {args.fif}")
        sys.exit(1)
    raw = mne.io.read_raw_fif(str(args.fif), preload=True, verbose=False)
    sfreq = raw.info['sfreq']
    events = mne.find_events(raw, stim_channel='STI 014', shortest_event=1)
    X_all, y_all = preprocess_epoch_data(raw, events, EPOCH_TMIN, EPOCH_TMAX)
    n_trials = len(y_all)
    print(f"Loaded {n_trials} trials from {args.fif.name}")

    # Wait for user to start
    start_msg = "Press Enter to begin playback"
    visual.TextStim(win, text=start_msg, height=40, color='black').draw()
    win.flip()
    event.waitKeys(keyList=['return'])
    fixation.draw(); win.flip(); core.wait(1.0)

    # Run feedback loop
    true_labels = []
    pred_labels = []
    # Flag to print adjustment only once
    printed_adjustment = False
    for i, (trial_X, trial_y) in enumerate(zip(X_all, y_all), start=1):
        # Prepare trial data
        X_trial = np.expand_dims(trial_X, axis=0)
        true_label = int(trial_y)
        # Display cue corresponding to this trial
        if args.game:
            scene.get_ready(true_label)            
        else:
            cue.text = 'LEFT' if true_label == 1 else 'RIGHT'
            cue.draw()
            win.flip()

        core.wait(EPOCH_TMAX)
        # Ensure input matches model's expected time dimension
        n_times_model = classifier.model.n_times
        t_current = X_trial.shape[2]
        if not printed_adjustment and t_current != n_times_model:
            print(f"[INFO] Adjusting trial time dimension from {t_current} to {n_times_model}")
            printed_adjustment = True
        if t_current > n_times_model:
            X_trial = X_trial[:, :, :n_times_model]
        elif t_current < n_times_model:
            # pad with zeros to the right
            pad_width = ((0,0), (0,0), (0, n_times_model - t_current))
            X_trial = np.pad(X_trial, pad_width, mode='constant', constant_values=0)
        # Subject index placeholder for CSP per-subject transform
        subject_indices = np.zeros(X_trial.shape[0], dtype=int)
        # Model inference
        res = classifier.evaluate(X_trial, np.array([true_label]), subject_indices=subject_indices)
        proba = res['probabilities'][0]
        # Now display feedback
        print(f"Trial {i}/{n_trials}: true={true_label}, proba={proba}")
        # Display feedback
        if args.game:
            scene.update(true_label, np.argmax(proba)+1, proba)
            core.wait(1.0)
            scene.baseline(args.delay)
        else:
            # Update and draw bars via shared BarFeedback
            bar_feedback.update(proba)
            bar_feedback.draw()
            fixation.draw()
            win.flip()
            core.wait(args.delay)
        # Record
        true_labels.append(true_label)
        pred_labels.append(np.argmax(proba)+1)

    # Summary
    accuracy = sum(1 for t,p in zip(true_labels, pred_labels) if t==p) / n_trials
    print(f"[RESULT] Playback accuracy: {accuracy*100:.2f}%")

    win.close()
    core.quit()


if __name__ == '__main__':
    main() 