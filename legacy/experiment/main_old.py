# acquisition + cues in one process ------------------------------------------
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from psychopy import visual, core
import numpy as np
import mne

params = BrainFlowInputParams(); 
params.serial_port = '/dev/cu.usbserial-DM0258IP'
board  = BoardShim(BoardIds.CYTON_DAISY_BOARD, params)
board.prepare_session(); board.start_stream()

win = visual.Window((800,600), fullscr=False, units='pix', color='white')
cue  = visual.TextStim(win, height=120, font='Helvetica', color='black')

n_trials = 1

for code, text in [(1,'LEFT'), (2,'RIGHT')]*n_trials:
    cue.text = text; cue.draw(); win.flip()
    board.insert_marker(code)           # <------------ precise marker!
    core.wait(4)                        # imagination period
    win.flip(); core.wait(2)

board.stop_stream()
data = board.get_board_data()

# instead, create and save an MNE Raw object
eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_DAISY_BOARD.value)
eeg_data = data[eeg_channels, :] / 1e6  # convert from uV to V
ch_names = BoardShim.get_eeg_names(BoardIds.CYTON_DAISY_BOARD.value)
sfreq = BoardShim.get_sampling_rate(BoardIds.CYTON_DAISY_BOARD.value)
ch_types = ['eeg'] * len(eeg_channels)
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
raw = mne.io.RawArray(eeg_data, info)
raw.save('session_raw.fif', overwrite=True)

board.release_session()
win.close()