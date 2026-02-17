import os
import numpy as np
import mne
import torch
from pathlib import Path

MU_BETA_BAND = (8, 30)       # Optimal frequency band for motor imagery
DEBUG = False                # Global debug flag to control printouts
PER_SUBJECT_NORMALIZATION = False

# Internal cache for loaded subject data to avoid repeated IO
_subject_data_cache = {}
# Cache of standard time length per (run_ids tuple, use_motor_channels)
_standard_length_cache = {}
# Cache for combined multi-subject loads to avoid repeated concatenation and slicing
_multi_subject_cache = {}

# Debug print function
def dprint(*args, **kwargs):
    """Only print if DEBUG is True"""
    if DEBUG:
        print(*args, **kwargs)

def load_subject_data(subject_id, run_ids, use_motor_channels):
    """
    Load data for a single subject with optional channel selection
    """

    # Check cache first
    key = (subject_id, tuple(run_ids), use_motor_channels)
    if key in _subject_data_cache:
        dprint(f"Using cached data for subject {subject_id}, runs {run_ids}, use_motor_channels={use_motor_channels}")
        return _subject_data_cache[key]
    
    # Try to load processed data
    processed_root = Path(os.environ.get("PUBLIC_PROCESSED_DATA_DIR", "data/public/processed_data"))
    all_epochs = []
    for run_id in run_ids:
        filepath = processed_root / f"sub-{subject_id}_run-{run_id}_processed-epo.fif"
        if not filepath.exists():
            raise FileNotFoundError(f"No processed data found at {filepath}")
        
        single_run_epochs = mne.read_epochs(str(filepath))
        all_epochs.append(single_run_epochs)
    
    # Combine epochs from all runs
    epochs = mne.concatenate_epochs(all_epochs)
    dprint(f"Loaded {len(epochs)} total epochs from {len(run_ids)} runs")
    
    # Select motor channels if requested
    if use_motor_channels:
        # Motor channels typically include C3, C1, Cz, C2, C4, CP3, CP1, CPz, CP2, CP4
        motor_channels = [ch for ch in epochs.ch_names if ch.startswith(('C', 'FC', 'CP'))]
        if len(motor_channels) > 0:
            dprint(f"Selecting {len(motor_channels)} motor-related channels")
            epochs = epochs.pick(motor_channels)
        else:
            dprint("No motor channels found, using all channels")
    
    # Apply bandpass filter for mu/beta rhythms
    epochs_filtered = epochs.copy().filter(
        l_freq=MU_BETA_BAND[0], 
        h_freq=MU_BETA_BAND[1], 
        method='fir',
        fir_window='hamming',
        verbose=False
    )
    dprint(f"Applied bandpass filter {MU_BETA_BAND[0]}-{MU_BETA_BAND[1]}Hz")
    
    # Find available event types
    motor_events = {}
    if 'TASK1T1' in epochs.event_id:
        motor_events['TASK1T1'] = epochs.event_id['TASK1T1']  # Left hand
    if 'TASK1T2' in epochs.event_id:
        motor_events['TASK1T2'] = epochs.event_id['TASK1T2']  # Right hand
    
    assert len(motor_events) == 2
    
    dprint(f"Classifying between: {list(motor_events.keys())}")
    
    # Select optimal time window for motor imagery (0.5-3.0s after cue)
    tmin, tmax = 0.0, 3.0
    epochs_cropped = epochs_filtered.copy().crop(tmin=tmin, tmax=tmax)
    
    # Extract data for the selected events
    selected_epochs = epochs_cropped[list(motor_events.keys())]
    
    # Get data and labels
    X = selected_epochs.get_data()
    y = selected_epochs.events[:, 2]

    # Scale EEG data from volts to microvolts for better numerical stability
    # Typical EEG amplitudes are 10-100 μV, so if data is in volts, values will be near zero
    if X.mean() < 0.001 and X.std() < 0.001:
        # Data appears to be in volts, scale to microvolts (×1,000,000)
        scaling_factor = 1e6
        X_scaled = X * scaling_factor
        dprint(f"Scaling EEG data by {scaling_factor}x (V to μV conversion)")
    else:
        # Data already in reasonable range, no scaling needed
        X_scaled = X
        dprint("No scaling needed, data already in appropriate range")
    
    # Convert event IDs to zero-indexed class labels
    unique_labels = np.unique(y)
    
    label_map = {label: i for i, label in enumerate(unique_labels)}
    
    y = np.array([label_map[label] for label in y])
    
    # dprint class distribution
    dprint(f"Class distribution: {np.bincount(y)} (total: {len(y)} trials)")
    
    # Cache the loaded data
    _subject_data_cache[key] = (X_scaled, y, epochs.info)
    
    return X_scaled, y, epochs.info


def load_multi_subject_data(subject_ids, run_ids, use_motor_channels=True, max_subjects=None):
    """
    Load and combine data from multiple subjects, ignoring those with non-standard time lengths
    
    Args:
        subject_ids: Specific subject IDs to load (None = all available)
        run_ids: Run IDs to use
        use_motor_channels: Whether to select only motor-related channels
        max_subjects: Maximum number of subjects to include (None = all)
        
    Returns:
        X_combined: Combined EEG data
        y_combined: Combined labels
        subject_indices_combined: Indices tracking which subject each trial comes from
    """
    if subject_ids is None:
        # Use all subjects by default (001-109)
        subject_ids = [f"{i:03d}" for i in range(1, 110)]
    
    # If max_subjects is specified, randomly sample subjects
    if max_subjects is not None and max_subjects < len(subject_ids):
        subject_ids = np.random.choice(subject_ids, size=max_subjects, replace=False)
        dprint(f"Randomly sampled {max_subjects} subjects: {', '.join(subject_ids)}")
    
    # Check if we've already loaded this multi-subject dataset
    multi_key = (frozenset(subject_ids), tuple(run_ids), use_motor_channels, max_subjects)
    if multi_key in _multi_subject_cache:
        dprint(f"Using cached multi-subject data for key {multi_key}")
        return _multi_subject_cache[multi_key]
    
    # Determine standard time length only once per (run_ids, use_motor_channels)
    len_key = (tuple(run_ids), use_motor_channels)
    if len_key in _standard_length_cache:
        standard_length = _standard_length_cache[len_key]
        dprint(f"Using cached standard time length of {standard_length} for runs {run_ids}")
    else:
        # Find the most common time dimension across subjects (sample up to 20)
        time_lengths = {}
        for subject_id in subject_ids[:min(40, len(subject_ids))]:
            try:
                X, _, _ = load_subject_data(subject_id, run_ids, use_motor_channels)
                _, _, length = X.shape
                time_lengths[length] = time_lengths.get(length, 0) + 1
            except Exception as e:
                dprint(f"Error loading subject {subject_id}: {str(e)}")
                continue
        if not time_lengths:
            raise ValueError("Could not determine time length from any subject")
        # Pick the most common time length
        standard_length = max(time_lengths.items(), key=lambda x: x[1])[0]
        dprint(f"Determined standard time length of {standard_length} (most common among sampled subjects)")
        _standard_length_cache[len_key] = standard_length
    
    # Process each subject
    all_X = []
    all_y = []
    all_subject_indices = []
    subject_count = 0
    skipped_count = 0
    
    dprint(f"Loading data from {len(subject_ids)} subjects...")
    
    for subject_id in subject_ids:
        try:
            X, y, _ = load_subject_data(subject_id, run_ids, use_motor_channels)
            
            # Skip subjects whose time dimension deviates from standard
            _, _, time_points = X.shape
            if time_points != standard_length:
                dprint(f"Subject {subject_id}: Skipped - non-standard time dimension ({time_points})")
                skipped_count += 1
                continue
                
            # Track which subject each trial came from
            subject_indices = np.full(len(y), subject_count)
            
            all_X.append(X)
            all_y.append(y)
            all_subject_indices.append(subject_indices)
            
            subject_count += 1
            
        except Exception as e:
            dprint(f"Error loading subject {subject_id}: {str(e)}")
            continue
    
    if not all_X:
        raise ValueError("No valid subject data was loaded")
    
    # Combine data from all subjects
    X_combined = np.concatenate(all_X, axis=0)
    y_combined = np.concatenate(all_y, axis=0)
    subject_indices_combined = np.concatenate(all_subject_indices, axis=0)
    
    # Cache combined data for fast reuse
    _multi_subject_cache[multi_key] = (X_combined, y_combined, subject_indices_combined)
    
    dprint(f"Combined dataset: {X_combined.shape[0]} trials, {X_combined.shape[1]} channels, {X_combined.shape[2]} timepoints")
    dprint(f"Class distribution: {np.bincount(y_combined)}")
    dprint(f"Used {subject_count} subjects, skipped {skipped_count} subjects with non-standard time lengths")
    
    return X_combined, y_combined, subject_indices_combined


class FeatureNormalizer:
    def __init__(self):
        self.eeg_mean = None  # torch.Tensor [1, C, 1]
        self.eeg_std = None   # torch.Tensor [1, C, 1]
        self.csp_mean = None  # torch.Tensor [1, D]
        self.csp_std = None   # torch.Tensor [1, D]
        
    def fit(self, X_eeg, X_csp=None):
        """Calculate normalization parameters from training data only using torch."""
        # Ensure torch tensor
        X_eeg_tensor = X_eeg if torch.is_tensor(X_eeg) else torch.tensor(X_eeg, dtype=torch.float32)
        # Compute per-channel mean and std over trials and time dims: shape [1, C, 1]
        self.eeg_mean = torch.mean(X_eeg_tensor, dim=(0, 2), keepdim=True)
        self.eeg_std = torch.std(X_eeg_tensor, dim=(0, 2), keepdim=True, unbiased=False)
        # Avoid zeros
        self.eeg_std = torch.where(self.eeg_std <= 1e-8, torch.ones_like(self.eeg_std), self.eeg_std)
        # CSP features normalization
        if X_csp is not None:
            X_csp_tensor = X_csp if torch.is_tensor(X_csp) else torch.tensor(X_csp, dtype=torch.float32)
            # mean/std across samples: shape [1, D]
            self.csp_mean = torch.mean(X_csp_tensor, dim=0, keepdim=True)
            self.csp_std = torch.std(X_csp_tensor, dim=0, keepdim=True, unbiased=False)
            self.csp_std = torch.where(self.csp_std <= 1e-8, torch.ones_like(self.csp_std), self.csp_std)

    def transform(self, X_eeg, X_csp=None):
        """Apply normalization using stored parameters with torch."""
        # Ensure torch tensor
        X_eeg_tensor = X_eeg if torch.is_tensor(X_eeg) else torch.tensor(X_eeg, dtype=torch.float32)
        self.eeg_mean = self.eeg_mean.to(X_eeg_tensor.device)
        self.eeg_std = self.eeg_std.to(X_eeg_tensor.device)
        # Standardize
        X_eeg_norm = (X_eeg_tensor - self.eeg_mean) / (self.eeg_std + 1e-8)
        X_eeg_norm = torch.clamp(X_eeg_norm, -5.0, 5.0)
        # CSP
        if X_csp is not None and self.csp_mean is not None:
            X_csp_tensor = X_csp if torch.is_tensor(X_csp) else torch.tensor(X_csp, dtype=torch.float32)
            self.csp_mean = self.csp_mean.to(X_csp_tensor.device)
            self.csp_std = self.csp_std.to(X_csp_tensor.device)
            X_csp_norm = (X_csp_tensor - self.csp_mean) / (self.csp_std + 1e-8)
        else:
            X_csp_norm = None
        return X_eeg_norm, X_csp_norm

    def save(self, path):
        # Save normalizer parameters
        params = {
            'eeg_mean': self.eeg_mean,
            'eeg_std': self.eeg_std,
            'csp_mean': self.csp_mean,
            'csp_std': self.csp_std
        }
        torch.save(params, path)
        
    @classmethod
    def load(cls, path):
        # Load saved normalizer
        params = torch.load(path)
        normalizer = cls()
        
        # Load basic parameters
        normalizer.eeg_mean = params['eeg_mean']
        normalizer.eeg_std = params['eeg_std']
        normalizer.csp_mean = params['csp_mean']
        normalizer.csp_std = params['csp_std']
            
        return normalizer