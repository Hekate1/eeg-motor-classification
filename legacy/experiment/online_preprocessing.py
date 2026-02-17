#!/usr/bin/env python3
import mne
import numpy as np


def preprocess_epoch_data(raw, events, tmin, tmax, notch_freq=60, l_freq=1.0, h_freq=50.0, z_threshold=2.0):
    """Extract and preprocess a single epoch from raw data for classification."""
    # Copy raw to avoid modifying the original data
    raw_copy = raw.copy()
    # Pre-filter and re-reference raw data
    raw_copy.notch_filter(freqs=notch_freq, picks="eeg", verbose=False)
    raw_copy.filter(l_freq=l_freq, h_freq=h_freq, picks="eeg", verbose=False)
    # Re-reference to average of all EEG channels
    raw_copy.set_eeg_reference('average', verbose=False)
    # Create Epochs for the specified events and time window
    epochs = mne.Epochs(
        raw_copy,
        events,
        event_id={"Left": 1, "Right": 2},
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        proj=False,
        picks="eeg",
        preload=True,
        verbose=False,
        on_missing='ignore'
    )
    # Drop bad epochs based on variance z-score
    data = epochs.get_data()
    variances = np.var(data, axis=(1, 2))
    z_scores = np.abs((variances - np.mean(variances)) / np.std(variances))
    bad_idx = np.where(z_scores > z_threshold)[0]
    if len(bad_idx) > 0:
        epochs.drop(bad_idx)
    # Extract processed data and labels
    X = epochs.get_data()
    y = epochs.events[:, 2]
    return X, y 