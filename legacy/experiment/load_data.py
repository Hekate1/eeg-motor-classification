import numpy as np
import mne
from brainflow.board_shim import BoardShim, BoardIds
import matplotlib.pyplot as plt

data = np.load("session.npy")
descr = BoardShim.get_board_descr(BoardIds.CYTON_DAISY_BOARD.value)

# ----- build an MNE Raw with only EEG + marker ------------------------------
eeg_rows   = descr['eeg_channels']          # [1 .. 16]
stim_row   = descr['marker_channel']        # 31
sfreq      = descr['sampling_rate']         # 125 Hz with Daisy

eeg_data = data[eeg_rows] * 1e-6            # µV → V
stim     = data[stim_row][None, :]

info = mne.create_info(
    ch_names=[f'Cyton{i+1}' for i in range(len(eeg_rows))] + ['STI 014'],
    sfreq=sfreq,
    ch_types=['eeg']*len(eeg_rows) + ['stim']
)

raw = mne.io.RawArray(np.vstack([eeg_data, stim]), info)
raw.set_montage('standard_1020', on_missing='ignore')  # optional
raw = raw.filter(l_freq=1, h_freq=60)
raw = raw.notch_filter(60)
raw.set_eeg_reference('average')

raw.plot(duration=10, scalings={'eeg': 200e-6, 'stim': 1})
raw.compute_psd().plot()                                 # spectral shape
events = mne.find_events(raw, stim_channel='STI 014') # should match cue count
mne.viz.plot_events(events, sfreq=sfreq)
plt.show()