from matplotlib import pyplot as plt
import numpy as np
import mne
import mne_icalabel

# Load data
file_path = "EEG Motor Movement:Imagery Dataset/sub-001/eeg/sub-001_task-motion_run-3_eeg.set"
raw = mne.io.read_raw_eeglab(file_path, preload=True)

# Copy data to modify
rawCopy = raw.copy()

# Notch filter at 60 Hz and highpass filter at 1 Hz
rawCopy.notch_filter(freqs=60)
rawCopy.filter(l_freq=1, h_freq=None)

# Remove bad channels and re-reference to average
rawCopy.info['bads'].extend(['Iz', 'Oz', 'O1', 'O2', 'Poz', 'Po3', 'Po4', 'Po7', 'Po8'])
rawCopy.set_eeg_reference('average')

# Apply ICA with # good channels - 1 components
# Extended infomax reccomended by icalabel
ica = mne.preprocessing.ICA(n_components=54, random_state=0, method='infomax', fit_params=dict(extended=True))
ica.fit(rawCopy)
labels = mne_icalabel.label_components(rawCopy, ica, method='iclabel')
ica.exclude = [idx for idx, label in enumerate(labels['labels']) if label != 'brain']
rawCopyWithICA = ica.apply(rawCopy.copy())

# IIR bandpass between 8 and 30 Hz
processed = rawCopyWithICA.copy().filter(l_freq=8, h_freq=30, method="iir")

################## End of Preprocessing ##################

# Epoching data from 0 to 1.5 seconds with baseline correction from -2 to 0 seconds
epochs = mne.Epochs(processed, tmin=-2, tmax=1.5, baseline=(-2,0), preload=True)
epochs.crop(tmin=0, tmax=1.5)

# Calculate PSD in alpha frequency range for left and right hand movements
left_epochs = epochs['TASK1T1']
right_epochs = epochs['TASK1T2']
alpha_band = (8, 13)
psd_left = left_epochs.compute_psd(fmin=alpha_band[0], fmax=alpha_band[1])
psd_right = right_epochs.compute_psd(fmin=alpha_band[0], fmax=alpha_band[1])

# Calculate average power during the epoch for each channel for left and right
psd_left_data, freqs_left = psd_left.get_data(return_freqs=True)
psd_right_data, freqs_right = psd_right.get_data(return_freqs=True)
psd_left_mean = psd_left_data.mean(axis=-1) # Shape: (epochs, channels)
psd_right_mean = psd_right_data.mean(axis=-1)
psd_left_avg = psd_left_mean.mean(axis=0) # Shape: (channels,)
psd_right_avg = psd_right_mean.mean(axis=0)

# Select relevant channels on left and right hemispheres
left_channels_of_interest = ["Fc1", "Fc3", "C1", "C3", "Cp1", "Cp3"]
right_channels_of_interest = ["Fc2", "Fc4", "C2", "C4", "Cp2", "Cp4"]
left_channel_indices = [epochs.ch_names.index(ch) for ch in left_channels_of_interest]
right_channel_indices = [epochs.ch_names.index(ch) for ch in right_channels_of_interest]

# Calculate mean alpha power for these channels
alpha_left_hand_left = psd_left_avg[left_channel_indices].mean()
alpha_left_hand_right = psd_left_avg[right_channel_indices].mean()
alpha_right_hand_left = psd_right_avg[left_channel_indices].mean()
alpha_right_hand_right = psd_right_avg[right_channel_indices].mean()

# Plot bar chart
conditions = ['Left Hand MI Left Brain', 'Left Hand MI Right Brain', 'Right Hand MI Left Brain', 'Right Hand MI Right Brain']
alpha_values = [alpha_left_hand_left, alpha_left_hand_right, alpha_right_hand_left, alpha_right_hand_right]

fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(conditions, alpha_values, color=['tab:blue','tab:orange','tab:blue','tab:orange'])
ax.set_ylabel('Mean Alpha Power')
ax.set_title('Mean Alpha Power in Sensorimotor Channels')
plt.show()