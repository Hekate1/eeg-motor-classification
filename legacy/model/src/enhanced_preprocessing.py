import numpy as np
import mne
import mne_icalabel
import os

def load_subject_data(file_path):
    # Load the .set file using mne
    return mne.io.read_raw_eeglab(file_path, preload=True)

def notch_highpass_rereference(data: mne.io.Raw, notch_freq=60, l_freq=1):
    dataCopy = data.copy()
    dataCopy.notch_filter(freqs=notch_freq, verbose=False)
    dataCopy.filter(l_freq=l_freq, h_freq=None, verbose=False)
    # Rereference the data to the average of all electrodes
    return dataCopy.set_eeg_reference('average', verbose=False)

def epoch_data(data: mne.io.Raw, e_start, e_end):
    return mne.Epochs(data, tmin=e_start, tmax=e_end, baseline=None, preload=True, verbose=False)

def bandpass_filter(data: mne.io.Raw, l_freq, h_freq):
    return data.copy().filter(l_freq=l_freq, h_freq=h_freq, method="fir", verbose=False)

def run_ica(data: mne.io.Raw, remove_labels, n_components=30, plot_sources=False, do_print=False):
    dataCopy = data.copy()
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=0, method='infomax', fit_params=dict(extended=True))
    ica.fit(dataCopy, verbose=False)

    labels = mne_icalabel.label_components(dataCopy, ica, method='iclabel')
    if do_print:
        print(labels)
    ica.exclude = [idx for idx, label in enumerate(labels['labels']) if label in remove_labels]
    if plot_sources:
        _ = ica.plot_sources(dataCopy)
    return ica.apply(dataCopy, verbose=False)

def auto_reject_channels(data: mne.io.Raw, z_threshold=3.0, do_print=False):
    """Automatically reject channels based on z-score of their variance."""
    dataCopy = data.copy()
    # Calculate variance for each channel
    variances = np.var(dataCopy.get_data(), axis=1)
    # Calculate z-scores
    z_scores = np.abs((variances - np.mean(variances)) / np.std(variances))
    # Find bad channels
    bad_channels = [ch for ch, z in zip(dataCopy.ch_names, z_scores) if z > z_threshold]
    dataCopy.info['bads'].extend(bad_channels)
    dataCopy.interpolate_bads()
    if do_print:
        print(f"Bad channels: {bad_channels}")
    return dataCopy

def auto_reject_trials(epochs: mne.Epochs, z_threshold=2.0, do_print=False):
    """Automatically reject trials based on z-score of their variance."""
    epochsCopy = epochs.copy()
    # Calculate variance for each trial
    variances = np.var(epochsCopy.get_data(), axis=(1, 2))
    # Calculate z-scores
    z_scores = np.abs((variances - np.mean(variances)) / np.std(variances))
    # Find bad trials
    bad_trials = np.where(z_scores > z_threshold)[0]
    epochsCopy.drop(bad_trials)
    if do_print:
        print(f"Bad trials: {bad_trials}")
    return epochsCopy

def save_processed_data(data: mne.Epochs, subject: str, run: int):
    data.save(f'processed_data/sub-{subject}_run-{run}_processed-epo.fif', overwrite=True)

def preprocess_data(subject: str, run: int, e_start: float = -1, e_end: float = 4, 
                   b_start: float = -1, b_end: float = 0):
    """Complete preprocessing pipeline with automatic channel and trial rejection."""
    # Construct file path
    file_path = f"EEG Motor Movement:Imagery Dataset/sub-{subject}/eeg/sub-{subject}_task-motion_run-{run}_eeg.set"
    
    data = notch_highpass_rereference(load_subject_data(file_path))
    data = auto_reject_channels(data)
    data = run_ica(data, ['eye blink', 'heart beat', 'muscle artifact', 'other'])
    data = bandpass_filter(data, 1, 50)
    data = epoch_data(data, e_start, e_end)
    data = auto_reject_trials(data)
    data = data.apply_baseline((b_start, b_end))
    
    return data

if __name__ == "__main__":
    # Batch preprocessing for multiple subjects and runs
    subjects = [f"{i:03d}" for i in range(1, 110)]  # 001 through 109
    runs = [3, 7, 11]  # Motor imagery runs
    
    # Create processed_data directory if it doesn't exist
    os.makedirs("processed_data", exist_ok=True)
    
    # Process each subject and run
    for subject in subjects:
        for run in runs:
            try:
                print(f"\n==== Processing subject {subject}, run {run} ====")
                processed_data = preprocess_data(subject, run)
                print(f"Number of epochs after preprocessing: {len(processed_data)}")
                print(f"Number of channels after preprocessing: {len(processed_data.ch_names)}")
                save_processed_data(processed_data, subject, run)
                print(f"Successfully saved processed data for subject {subject}, run {run}")
            except Exception as e:
                print(f"Error processing subject {subject}, run {run}: {str(e)}")
                continue