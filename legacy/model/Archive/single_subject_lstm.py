"""
Single Subject EEG Classification with LSTM

This script classifies EEG motor imagery data for a single subject using an LSTM network.
It implements significant data augmentation techniques to improve performance.
"""

import os
import numpy as np
import mne
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

# Global configuration
SUBJECT_ID = "001"           # Single subject to focus on
RUN_IDS = ["3", "7", "11"]   # Motor imagery runs
RANDOM_SEED = 42             # For reproducibility
USE_MOTOR_CHANNELS = True    # Whether to select only motor-related channels
MU_BETA_BAND = (8, 30)       # Optimal frequency band for motor imagery

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Set device for PyTorch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_subject_data(subject_id=SUBJECT_ID, run_ids=RUN_IDS, use_motor_channels=USE_MOTOR_CHANNELS):
    """
    Load data for a single subject with optional channel selection
    """
    try:
        # Try to load processed data
        all_epochs = []
        for run_id in run_ids:
            filepath = f'processed_data/sub-{subject_id}_run-{run_id}_processed-epo.fif'
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"No processed data found at {filepath}")
            
            epochs = mne.read_epochs(filepath)
            all_epochs.append(epochs)
        
        # Combine epochs from all runs
        epochs = mne.concatenate_epochs(all_epochs)
        print(f"Loaded {len(epochs)} total epochs from {len(run_ids)} runs")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Creating sample data for demonstration purposes...")
        # Create sample data with typical dimensions for motor imagery
        n_channels = 64
        n_times = 640  # 4 seconds at 160Hz
        n_trials = 100  # 50 per class
        
        # Generate random EEG-like data
        X_sample = np.random.randn(n_trials, n_channels, n_times) * 0.5
        
        # Create labels (50 trials per class for binary classification)
        y_sample = np.concatenate([np.zeros(n_trials//2), np.ones(n_trials//2)])
        
        # Create info for MNE epochs
        ch_names = [f'EEG{i:03d}' for i in range(1, n_channels+1)]
        ch_types = ['eeg'] * n_channels
        info = mne.create_info(ch_names=ch_names, sfreq=160, ch_types=ch_types)
        
        # Create events array (required for MNE epochs)
        events = np.zeros((n_trials, 3), dtype=int)
        events[:, 0] = np.arange(0, n_trials * 100, 100)  # Event times
        events[:, 2] = y_sample + 1  # Event IDs (1 or 2)
        
        # Create MNE epochs with the generated data
        epochs = mne.EpochsArray(X_sample, info, events=events, tmin=0)
        epochs.event_id = {'TASK1T1': 1, 'TASK1T2': 2}  # Standard motor imagery event IDs
        
        print(f"Created sample data with {n_trials} trials, {n_channels} channels, and {n_times} timepoints")
    
    # Select motor channels if requested
    if use_motor_channels:
        # Motor channels typically include C3, C1, Cz, C2, C4, CP3, CP1, CPz, CP2, CP4
        motor_channels = [ch for ch in epochs.ch_names if ch.startswith(('C', 'FC', 'CP'))]
        if len(motor_channels) > 0:
            print(f"Selecting {len(motor_channels)} motor-related channels")
            epochs.pick_channels(motor_channels)
        else:
            print("No motor channels found, using all channels")
    
    # Apply bandpass filter for mu/beta rhythms
    epochs_filtered = epochs.copy().filter(
        l_freq=MU_BETA_BAND[0], 
        h_freq=MU_BETA_BAND[1], 
        method='fir',
        fir_window='hamming',
        verbose=False
    )
    print(f"Applied bandpass filter {MU_BETA_BAND[0]}-{MU_BETA_BAND[1]}Hz")
    
    # Find available event types
    motor_events = {}
    if 'TASK1T1' in epochs.event_id:
        motor_events['TASK1T1'] = epochs.event_id['TASK1T1']  # Left hand
    if 'TASK1T2' in epochs.event_id:
        motor_events['TASK1T2'] = epochs.event_id['TASK1T2']  # Right hand
    
    if len(motor_events) < 2:
        # Fallback if standard event IDs aren't found
        event_ids = list(epochs.event_id.values())
        if len(event_ids) >= 2:
            motor_events = {name: id for name, id in epochs.event_id.items()}
        else:
            raise ValueError("Could not find at least two event types in the data")
    
    print(f"Classifying between: {list(motor_events.keys())}")
    
    # Select optimal time window for motor imagery (0.5-3.0s after cue)
    tmin, tmax = 0.5, 3.0
    epochs_cropped = epochs_filtered.copy().crop(tmin=tmin, tmax=tmax)
    
    # Extract data for the selected events
    selected_epochs = epochs_cropped[list(motor_events.keys())]
    
    # Get data and labels
    X = selected_epochs.get_data()
    y = selected_epochs.events[:, 2]
    
    # Convert event IDs to zero-indexed class labels
    unique_labels = np.unique(y)
    print(f"Original event IDs: {unique_labels}")
    
    label_map = {label: i for i, label in enumerate(unique_labels)}
    print(f"Label mapping: {label_map}")
    
    y = np.array([label_map[label] for label in y])
    
    # Print class distribution
    print(f"Class distribution: {np.bincount(y)} (total: {len(y)} trials)")
    
    # Verify data characteristics
    print(f"X shape: {X.shape}, X mean: {X.mean():.4f}, X std: {X.std():.4f}")
    print(f"X min: {X.min():.4f}, X max: {X.max():.4f}")
    
    return X, y, epochs.info


class EEGDataAugmenter:
    """
    Class for applying data augmentation techniques to EEG data
    """
    def __init__(self, noise_level=0.03, scale_range=(0.9, 1.1),
                 chunk_max_offset=30):
        """
        Initialize the data augmenter with various augmentation parameters
        """
        self.noise_level = noise_level  # Reduced from 0.05 to 0.03
        self.scale_range = scale_range  # Reduced from (0.8, 1.2) to (0.9, 1.1)
        self.chunk_max_offset = chunk_max_offset  # Reduced from 50 to 30
    
    def add_noise(self, x):
        """Add random Gaussian noise to the signal"""
        # Scale noise by the standard deviation of the signal for better adaptation
        noise_scale = self.noise_level * np.std(x)
        noise = np.random.normal(0, noise_scale, x.shape)
        return x + noise
    
    def scale_amplitude(self, x):
        """Scale the amplitude of the signal randomly"""
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        return x * scale
    
    def chunk_selection(self, x):
        """
        Select a continuous chunk of the trial with a random offset
        This preserves temporal relationships while providing variation
        """
        n_channels, n_times = x.shape
        
        # Determine maximum offset based on signal length
        max_offset = min(self.chunk_max_offset, n_times // 6)  # Reduced from 4 to 6
        
        # Generate random offset (start point)
        offset = np.random.randint(0, max_offset + 1)
        
        # Select chunk from the original signal
        if offset > 0:
            # Create a new array with the same shape
            x_new = np.zeros_like(x)
            
            # Fill with the selected chunk, padding with zeros
            x_new[:, :-offset] = x[:, offset:]
            
        return x_new
        else:
            # No offset, return original
            return x.copy()
    
    def variable_length_sequence(self, x):
        """
        Create a variable length sequence by randomly trimming the end
        and/or beginning of the signal with tapering to avoid abrupt changes
        """
        n_channels, n_times = x.shape
        
        # Don't trim more than 15% from each end (reduced from 25%)
        max_trim = int(n_times * 0.15)
        
        # Random amount to trim from start and end
        start_trim = np.random.randint(0, max_trim)
        end_trim = np.random.randint(0, max_trim)
        
        # Create output array (same size, will be padded with zeros)
        x_new = np.zeros_like(x)
        
        # Extract trimmed sequence
        trimmed_seq = x[:, start_trim:n_times-end_trim]
        
        # Place in center of output array
        trim_length = trimmed_seq.shape[1]
        start_idx = (n_times - trim_length) // 2
        
        # Apply tapering window to avoid abrupt transitions
        taper_length = min(20, trim_length // 4)
        taper_in = np.linspace(0, 1, taper_length)
        taper_out = np.linspace(1, 0, taper_length)
        
        for c in range(n_channels):
            # Apply the main part of the signal
            x_new[c, start_idx:start_idx+trim_length] = trimmed_seq[c]
            
            # Apply tapering to beginning and end of the trimmed sequence
            if taper_length > 0:
                x_new[c, start_idx:start_idx+taper_length] *= taper_in
                x_new[c, start_idx+trim_length-taper_length:start_idx+trim_length] *= taper_out
        
        return x_new
    
    def frequency_band_noise(self, x):
        """
        Add noise only in specific frequency bands relevant to motor imagery
        This preserves the overall signal structure while adding variability
        to relevant frequency components
        """
        n_channels, n_times = x.shape
        x_new = x.copy()
        
        # FFT parameters
        fs = 160  # Assuming 160Hz sampling rate (typical for EEG)
        
        for c in range(n_channels):
            # Convert to frequency domain
            x_fft = np.fft.rfft(x[c])
            freqs = np.fft.rfftfreq(n_times, d=1/fs)
            
            # Target mu (8-12Hz) and beta (13-30Hz) bands for motor imagery
            mu_mask = (freqs >= 8) & (freqs <= 12)
            beta_mask = (freqs >= 13) & (freqs <= 30)
            
            # Add noise only to these frequency bands
            mu_noise = np.random.normal(0, 0.1, size=sum(mu_mask)) * np.abs(x_fft[mu_mask].mean())
            beta_noise = np.random.normal(0, 0.05, size=sum(beta_mask)) * np.abs(x_fft[beta_mask].mean())
            
            # Apply noise
            x_fft[mu_mask] = x_fft[mu_mask] * (1 + mu_noise)
            x_fft[beta_mask] = x_fft[beta_mask] * (1 + beta_noise)
            
            # Convert back to time domain
            x_new[c] = np.fft.irfft(x_fft, n=n_times)
            
        return x_new
    
    def mild_temporal_warp(self, x):
        """
        Apply a mild non-linear temporal warping to simulate
        slight variations in timing of neural responses
        """
        n_channels, n_times = x.shape
        result = np.zeros_like(x)
        
        # Create a smooth warping function using a sinusoid
        # This will alternately stretch and compress different parts of the signal
        warp_amount = np.random.uniform(0.03, 0.07)  # 3-7% warping
        warp = np.sin(np.linspace(0, 2*np.pi, n_times)) * warp_amount
        
        # Create the warping indices
        indices = np.arange(n_times)
        warped_indices = indices * (1 + warp)
        
        # Ensure indices are within bounds
        warped_indices = np.clip(warped_indices, 0, n_times - 1)
        
        # Apply warping to each channel
        for c in range(n_channels):
            # Interpolate the signal according to the warping
            result[c] = np.interp(warped_indices, indices, x[c])
            
        return result
    
    def augment(self, x):
        """Apply multiple augmentation techniques to a single trial"""
        # Updated list of augmentations
        augmentations = [
            self.add_noise,
            self.scale_amplitude,
            self.chunk_selection,
            self.variable_length_sequence,
            self.frequency_band_noise,
            self.mild_temporal_warp
        ]
        
        # Apply 2-3 random augmentations
        num_augmentations = np.random.randint(2, 4)
        selected_augmentations = np.random.choice(augmentations, 
                                                 size=num_augmentations, 
                                                 replace=False)
        
        # Apply selected augmentations
        x_aug = x.copy()
        for augmentation in selected_augmentations:
            x_aug = augmentation(x_aug)
            
        return x_aug

    def augment_batch(self, X, y, augmentation_factor=2):
        """Generate augmented data for a batch of EEG trials"""
        n_trials = X.shape[0]
        augmented_X = []
        augmented_y = []
        
        # Include original data
        augmented_X.append(X)
        augmented_y.append(y)
        
        # Add augmented data
        for _ in range(augmentation_factor):
            X_batch_aug = np.array([self.augment(X[i]) for i in range(n_trials)])
            augmented_X.append(X_batch_aug)
            augmented_y.append(y)
        
        # Concatenate original and augmented data
        X_augmented = np.concatenate(augmented_X, axis=0)
        y_augmented = np.concatenate(augmented_y, axis=0)
        
        return X_augmented, y_augmented


class FrequencyTimeLSTMModel(nn.Module):
    """
    Dual-path model that combines frequency domain features with time domain features
    for improved EEG classification
    """
    def __init__(self, n_channels, n_classes, hidden_size=64, dropout=0.5, bidirectional=False):
        super(FrequencyTimeLSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.n_directions = 2 if bidirectional else 1
        
        # Time-domain path
        self.time_feature_extractor = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(16),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(kernel_size=2, stride=2)  # Reduce sequence length
        )
        
        # Frequency-domain path
        self.freq_feature_extractor = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ELU(),
            nn.Dropout(dropout)
        )
        
        # Separate frequency band extractors for mu and beta rhythms
        self.mu_extractor = nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1)
        self.beta_extractor = nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1)
        
        # Time-domain LSTM
        self.lstm = nn.LSTM(
            input_size=16,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Frequency domain dense layers
        self.freq_processor = nn.Sequential(
            nn.Linear(16 * 2, hidden_size),  # 16 = 8 (mu) + 8 (beta)
            nn.Dropout(dropout),
            nn.ELU()
        )
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * (self.n_directions + 1), hidden_size),
            nn.Dropout(dropout),
            nn.ELU(),
            nn.Linear(hidden_size, n_classes)
        )
        
        # Weight initialization
        self._init_weights()
        
    def _init_weights(self):
        """Apply suitable weight initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                elif 'conv' in name:
                    nn.init.kaiming_normal_(param, nonlinearity='relu')
                else:
                    # Check parameter dimensions before applying Xavier init
                    if len(param.shape) >= 2:
                    nn.init.xavier_normal_(param)
                else:
                    # For 1D weights (e.g., in batch norm), use constant initialization
                    nn.init.constant_(param, 1.0)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def extract_frequency_features(self, x):
        """
        Extract frequency domain features from the EEG signal
        focusing on mu (8-12 Hz) and beta (13-30 Hz) bands
        """
        batch_size, n_channels, n_times = x.size()
        
        # Apply FFT to get frequency representation
        # Use real FFT since EEG signals are real-valued
        x_fft = torch.fft.rfft(x, dim=2)
        
        # Get magnitudes (absolute values of complex numbers)
        x_fft_mag = torch.abs(x_fft)
        
        # Process with convolutional layers
        freq_features = self.freq_feature_extractor(x_fft_mag)
        
        # Extract mu and beta rhythm features
        # Assuming sampling rate of 160Hz, frequency bins are ~0.25Hz apart in a ~4s signal
        # Mu rhythm: 8-12 Hz (bins 32-48)
        # Beta rhythm: 13-30 Hz (bins 52-120)
        
        # Use the whole spectrum but let convolution focus on relevant bands
        mu_features = self.mu_extractor(freq_features)
        beta_features = self.beta_extractor(freq_features)
        
        # Global average pooling across frequency dimension
        mu_features = torch.mean(mu_features, dim=2)    # Shape: [batch_size, 8]
        beta_features = torch.mean(beta_features, dim=2)  # Shape: [batch_size, 8]
        
        # Print shapes for debugging
        # print(f"Mu features shape: {mu_features.shape}, Beta features shape: {beta_features.shape}")
        
        # Concatenate mu and beta features
        freq_features_combined = torch.cat((mu_features, beta_features), dim=1)  # Should be [batch_size, 16]
        
        # Double the features to match the expected input of freq_processor (32)
        # This is a simple fix to match dimensions without changing the model architecture
        freq_features_combined = torch.cat((freq_features_combined, freq_features_combined), dim=1)  # Now [batch_size, 32]
        
        return freq_features_combined

    def forward(self, x):
        """
        Forward pass through both time and frequency domain pathways
        """
        batch_size = x.size(0)
        
        # === Time-domain pathway ===
        time_features = self.time_feature_extractor(x)  # (batch_size, 16, n_times/2)
        
        # Transpose for LSTM: (batch_size, n_times/2, 16)
        time_features = time_features.permute(0, 2, 1)
        
        # Apply LSTM
        lstm_out, _ = self.lstm(time_features)  # (batch_size, n_times/2, hidden_size*n_directions)
        
        # Use last time step output for classification
        time_features_final = lstm_out[:, -1, :]
        
        # === Frequency-domain pathway ===
        freq_features = self.extract_frequency_features(x)
        freq_features_processed = self.freq_processor(freq_features)
        
        # === Combine features from both pathways ===
        combined_features = torch.cat((time_features_final, freq_features_processed), dim=1)
        
        # Apply classifier
        output = self.classifier(combined_features)
        
        return output


class LSTMClassifier:
    """
    LSTM classifier with training, evaluation, and prediction functionality
    """
    def __init__(self, n_classes, hidden_size=128, num_layers=2, dropout=0.3, 
                 bidirectional=True, lr=0.0005, batch_size=8, n_epochs=300, 
                 weight_decay=0.0001, device=None, augmentation_factor=3):
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.weight_decay = weight_decay
        self.device = device if device is not None else DEVICE
        self.augmentation_factor = augmentation_factor
        
        # Data augmenter
        self.augmenter = EEGDataAugmenter()
        
        # Model will be initialized when data dimensions are known
        self.model = None
        
        print(f"Using device: {self.device}")
    
    def save_model(self, filepath):
        """Save the trained model and configuration"""
        if self.model is None:
            raise ValueError("Model must be trained before saving")
        
        # Create a dictionary with model state and configuration
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'config': {
                'n_classes': self.n_classes,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'bidirectional': self.bidirectional,
                'model_params': {
                    name: param.shape for name, param in self.model.named_parameters()
                }
            }
        }
        
        # Save the model
        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath, device=None):
        """Load a trained model from a file"""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        # Load the saved dictionary
        save_dict = torch.load(filepath, map_location=device)
        
        # Extract configuration
        config = save_dict['config']
        
        # Create an instance of the classifier
        classifier = cls(
            n_classes=config['n_classes'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            bidirectional=config['bidirectional'],
            device=device
        )
        
        # Get the input dimensions from the first layer
        first_param_name = next(iter(config['model_params']))
        n_channels = None
        for name, shape in config['model_params'].items():
            if 'input_bn.weight' in name:
                n_channels = shape[0]
                break
        
        if n_channels is None:
            raise ValueError("Could not determine input dimensions from saved model")
            
        # Initialize the model with the saved dimensions
        classifier.model = FrequencyTimeLSTMModel(
            n_channels=n_channels,
            n_classes=config['n_classes'],
            hidden_size=config['hidden_size'],
            dropout=config['dropout'],
            bidirectional=config['bidirectional']
        ).to(device)
        
        # Load the state dictionary
        classifier.model.load_state_dict(save_dict['model_state_dict'])
        
        print(f"Model loaded from {filepath}")
        return classifier
    
    def evaluate_on_data(self, X, y, subject_indices=None):
        """Evaluate the model on new data with subject-specific normalization"""
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
            
        # Apply subject-specific normalization
        print("Applying subject-specific normalization for evaluation...")
        X_scaled = subject_specific_normalize(X, subject_indices)
        
        # Check for any NaN or inf values after normalization
        if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
            print("WARNING: Found NaN or Inf values after normalization. Replacing with zeros.")
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        # Evaluate
        self.model.eval()
        all_targets = []
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        cm = confusion_matrix(all_targets, all_predictions)
        
        # Detailed per-class metrics
        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions)
        
        class_accuracies = {}
        for i in range(self.n_classes):
            class_correct = np.sum((all_targets == i) & (all_predictions == i))
            class_total = np.sum(all_targets == i)
            if class_total > 0:
                class_accuracies[i] = class_correct / class_total
            else:
                class_accuracies[i] = float('nan')
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': np.array(all_probabilities),
            'class_accuracies': class_accuracies
        }

    def _compute_class_weights(self, y):
        """Compute class weights for imbalanced datasets"""
        class_counts = np.bincount(y)
        total_samples = len(y)
        n_classes = len(class_counts)
        
        # Compute weights inversely proportional to class frequencies
        # Use a stronger weighting factor (square root instead of linear)
        weights = np.sqrt(total_samples / (n_classes * class_counts))
        # Normalize weights
        weights = weights / weights.sum() * n_classes
        
        print(f"Detailed class weights calculation:")
        print(f"  Class counts: {class_counts}")
        print(f"  Raw inverse weights: {total_samples / (n_classes * class_counts)}")
        print(f"  Square root weights: {weights}")
        
        return torch.FloatTensor(weights).to(self.device)
    
    def _prepare_data(self, X, y, val_split=0.2, apply_augmentation=True):
        """Prepare data for training with optional augmentation"""
        # Print original class distribution
        print(f"Original class distribution: {np.bincount(y)}")
        
        # First split into training and validation sets - use stratified sampling
        # Create indices for each class
        indices_class_0 = [i for i, label in enumerate(y) if label == 0]
        indices_class_1 = [i for i, label in enumerate(y) if label == 1]
        
        # Shuffle indices
        np.random.shuffle(indices_class_0)
        np.random.shuffle(indices_class_1)
        
        # Calculate validation sizes for each class
        val_size_0 = int(len(indices_class_0) * val_split)
        val_size_1 = int(len(indices_class_1) * val_split)
        
        # Split indices
        train_indices = indices_class_0[val_size_0:] + indices_class_1[val_size_1:]
        val_indices = indices_class_0[:val_size_0] + indices_class_1[:val_size_1]
        
        # Split the data
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_val = X[val_indices]
        y_val = y[val_indices]
        
        # Apply class-aware augmentation to training data
        if apply_augmentation:
            # Determine which class is minority
            class_counts = np.bincount(y_train)
            if len(class_counts) >= 2:
                minority_class = np.argmin(class_counts)
                majority_class = np.argmax(class_counts)
                
                # Extract samples by class
                minority_indices = np.where(y_train == minority_class)[0]
                majority_indices = np.where(y_train == majority_class)[0]
                
                minority_samples = X_train[minority_indices]
                minority_labels = y_train[minority_indices]
                
                print(f"Class distribution before augmentation - Class 0: {np.sum(y_train == 0)}, Class 1: {np.sum(y_train == 1)}")
                print(f"Performing class-aware augmentation, focusing on minority class {minority_class}")
                
                # Calculate how many augmentations to perform for minority class
                # Generate more minority samples to balance the dataset
                target_count = len(majority_indices) * 2  # Double the majority class size
                augmentation_factor = max(2, int(target_count / len(minority_indices)))
                
                print(f"Augmenting minority class {minority_class} with factor {augmentation_factor}")
                
                # Augment only the minority class with a higher factor
                X_minority_aug, y_minority_aug = self.augmenter.augment_batch(
                    minority_samples, 
                    minority_labels, 
                    augmentation_factor
                )
                
                # Augment majority class with a lower factor
                X_majority_aug, y_majority_aug = self.augmenter.augment_batch(
                    X_train[majority_indices], 
                    y_train[majority_indices], 
                    1  # Only augment by 1x for majority class
                )
                
                # Combine original data with augmented data
                X_train = np.concatenate([X_train, X_minority_aug, X_majority_aug], axis=0)
                y_train = np.concatenate([y_train, y_minority_aug, y_majority_aug], axis=0)
                
                print(f"Applied data augmentation: {len(train_indices)} trials → {X_train.shape[0]} trials")
                print(f"Class distribution after augmentation - Class 0: {np.sum(y_train == 0)}, Class 1: {np.sum(y_train == 1)}")
            else:
                # Fallback to regular augmentation if we don't have both classes
                X_augmented, y_augmented = self.augmenter.augment_batch(X_train, y_train, self.augmentation_factor)
                print(f"Applied regular data augmentation: {X_train.shape[0]} trials → {X_augmented.shape[0]} trials")
                X_train, y_train = X_augmented, y_augmented
            
            # Print augmented class distribution
            print(f"Augmented training class distribution: {np.bincount(y_train)}")
        
        # Apply subject-specific normalization
        # Both training and validation sets are normalized separately, as they would be in a real scenario
        print("Applying subject-specific normalization (single subject)...")
        X_train = subject_specific_normalize(X_train)
        X_val = subject_specific_normalize(X_val)
        
        print(f"Normalization results - Training mean: {X_train.mean():.4f}, std: {X_train.std():.4f}")
        print(f"Validation mean: {X_val.mean():.4f}, std: {X_val.std():.4f}")
        
        # Check for any NaN or Inf values after normalization
        if np.isnan(X_train).any() or np.isinf(X_train).any():
            print("WARNING: Found NaN or Inf values in training data after normalization. Replacing with zeros.")
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        if np.isnan(X_val).any() or np.isinf(X_val).any():
            print("WARNING: Found NaN or Inf values in validation data after normalization. Replacing with zeros.")
            X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        print(f"Created stratified split: {len(train_indices)} original training samples, {X_train.shape[0]} after augmentation, {len(val_indices)} validation samples")
        
        return train_loader, val_loader
    
    def train_and_evaluate(self, X, y, val_split=0.2, early_stopping_patience=50):
        """Train and evaluate the LSTM model"""
        n_channels, n_times = X.shape[1], X.shape[2]
        
        # Initialize model
        self.model = FrequencyTimeLSTMModel(
            n_channels=n_channels,
            n_classes=self.n_classes,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        ).to(self.device)
        
        # Print model summary
        print(f"Frequency-Time LSTM model architecture:\n{self.model}")
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params:,}")
        
        # Prepare data with augmentation
        train_loader, val_loader = self._prepare_data(X, y, val_split, apply_augmentation=True)
        
        # Compute class weights for imbalanced datasets
        class_weights = self._compute_class_weights(y)
        print(f"Class weights: {class_weights}")
        
        # Implementation of focal loss to address class imbalance
        class FocalLoss(nn.Module):
            def __init__(self, weight=None, gamma=2.0, reduction='mean'):
                super(FocalLoss, self).__init__()
                self.weight = weight
                self.gamma = gamma
                self.reduction = reduction
                self.ce_loss = nn.CrossEntropyLoss(weight=weight, reduction='none')
                
            def forward(self, inputs, targets):
                ce_loss = self.ce_loss(inputs, targets)
                pt = torch.exp(-ce_loss)
                focal_loss = (1 - pt) ** self.gamma * ce_loss
                
                if self.reduction == 'mean':
                    return focal_loss.mean()
                elif self.reduction == 'sum':
                    return focal_loss.sum()
                else:
                    return focal_loss
        
        # Use focal loss instead of regular cross-entropy
        criterion = FocalLoss(
            weight=class_weights,
            gamma=2.0  # Higher gamma gives more weight to hard-to-classify examples
        )
        
        # Optimizer with learning rate warmup and weight decay
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler with longer warmup
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr * 3,  # Reduced peak learning rate
            steps_per_epoch=len(train_loader),
            epochs=self.n_epochs,
            pct_start=0.3,  # Longer warmup period
            div_factor=10,  # Smaller initial learning rate
            final_div_factor=100
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': [],
            'class_preds': []  # Track class prediction distributions
        }
        
        # Early stopping variables
        best_val_loss = float('inf')
        best_val_balanced_acc = 0.0  # Track balanced accuracy for early stopping
        patience_counter = 0
        best_model_state = None
        best_epoch = 0
        no_improvement_epochs = 0
        
        # Create a tensor to record all validation predictions for debugging
        all_val_targets_history = []
        all_val_preds_history = []
        
        # Variables for more gradual weight adjustment
        current_class_weights = class_weights.clone()
        weight_adjustment_rate = 0.1  # More gradual adjustment rate (was essentially 1.0 before)
        consecutive_imbalanced_epochs = 0
        max_adjustment_per_epoch = 0.2  # Maximum change in weight per epoch
        
        # Print interruption instructions
        print("\nTraining started. Press Ctrl+C to manually stop training and save the best model so far.")
        
        try:
        # Training loop
        for epoch in range(self.n_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # Track predictions for each class in training
            train_class_correct = [0] * self.n_classes
            train_class_total = [0] * self.n_classes
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Reduced grad clipping
                optimizer.step()
                scheduler.step()
                
                # Track loss and accuracy
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += targets.size(0)
                train_correct += (predicted == targets).sum().item()
                
                # Track per-class accuracy
                for i in range(len(targets)):
                    label = targets[i].item()
                    pred = predicted[i].item()
                    train_class_total[label] += 1
                    if label == pred:
                        train_class_correct[label] += 1
            
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            
            # Calculate per-class accuracy for training
            train_class_acc = [
                train_class_correct[i] / max(1, train_class_total[i]) 
                for i in range(self.n_classes)
            ]
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            # Track predictions for each class in validation
            val_class_correct = [0] * self.n_classes
            val_class_total = [0] * self.n_classes
            all_val_targets = []
            all_val_preds = []
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    
                    # Track loss and accuracy
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += targets.size(0)
                    val_correct += (predicted == targets).sum().item()
                    
                    # Store targets and predictions for debugging
                    all_val_targets.extend(targets.cpu().numpy())
                    all_val_preds.extend(predicted.cpu().numpy())
                    
                    # Track per-class accuracy
                    for i in range(len(targets)):
                        label = targets[i].item()
                        pred = predicted[i].item()
                        val_class_total[label] += 1
                        if label == pred:
                            val_class_correct[label] += 1
                
                val_loss = val_loss / val_total
                val_acc = val_correct / val_total
                
                # Calculate per-class accuracy for validation
                val_class_acc = [
                    val_class_correct[i] / max(1, val_class_total[i]) 
                    for i in range(self.n_classes)
                ]
                    
                    # Calculate balanced accuracy for validation (average of per-class accuracies)
                    val_balanced_acc = sum(val_class_acc) / len(val_class_acc)
                
                # Save all targets and predictions for this epoch
                all_val_targets_history.append(all_val_targets)
                all_val_preds_history.append(all_val_preds)
            
            # Get current learning rate
            current_lr = scheduler.get_last_lr()[0]
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['learning_rates'].append(current_lr)
            
            # Calculate prediction distribution for debugging
            val_pred_counts = np.bincount(all_val_preds, minlength=self.n_classes)
            val_pred_percentages = val_pred_counts / len(all_val_preds)
            history['class_preds'].append(val_pred_percentages)
            
            # Print progress with per-class accuracy
            print(f'Epoch {epoch+1}/{self.n_epochs}: '
                  f'train_loss={train_loss:.4f}, train_acc={train_acc:.4f} '
                  f'(class acc: {[f"{acc:.2f}" for acc in train_class_acc]}), '
                  f'val_loss={val_loss:.4f}, val_acc={val_acc:.4f} '
                  f'(class acc: {[f"{acc:.2f}" for acc in val_class_acc]}), '
                      f'balanced_acc={val_balanced_acc:.4f}, '
                  f'lr={current_lr:.6f}')
            
            # Print class prediction distribution
            print(f'  Validation class distribution: targets {np.bincount(all_val_targets)}, '
                  f'predictions {np.bincount(all_val_preds)}')
            
                # Check if model is predicting the same class for most samples (more than 80%)
                most_predicted_class = np.argmax(val_pred_percentages)
                most_predicted_percentage = val_pred_percentages[most_predicted_class]
                
                if most_predicted_percentage > 0.8:
                    print(f"  ! Warning: Model is predicting {most_predicted_class} for "
                          f"{most_predicted_percentage*100:.1f}% of samples")
                    
                    # More gradual weight adjustment
                    consecutive_imbalanced_epochs += 1
                    
                    # Only adjust weights after seeing imbalance for several epochs
                    if consecutive_imbalanced_epochs >= 3 and epoch > 10:
                        # Calculate adjustment factor (more gradual)
                        adjustment_factor = min(max_adjustment_per_epoch, weight_adjustment_rate * consecutive_imbalanced_epochs)
                        
                        # Create weight adjustments
                        new_weights = current_class_weights.clone()
                        
                        # Decrease weight for dominant class, increase for others
                        for i in range(self.n_classes):
                            if i == most_predicted_class:
                                # Decrease weight for dominant class
                                new_weights[i] = max(0.3, current_class_weights[i] * (1 - adjustment_factor))
                            else:
                                # Increase weight for non-dominant classes
                                new_weights[i] = min(3.0, current_class_weights[i] * (1 + adjustment_factor))
                        
                        # Normalize weights to keep the same scale
                        new_weights = new_weights / new_weights.sum() * self.n_classes
                        
                        # Update criterion with new weights
                        print(f"  → Adjusted class weights: {current_class_weights} → {new_weights} (adjustment: {adjustment_factor:.3f})")
                        current_class_weights = new_weights
                        
                        criterion = FocalLoss(
                            weight=current_class_weights,
                            gamma=2.0
                        )
                else:
                    # Reset consecutive imbalanced epochs counter when prediction is balanced
                    consecutive_imbalanced_epochs = 0
                
                # Early stopping check based on balanced accuracy instead of just loss
                improved_balanced_acc = val_balanced_acc > best_val_balanced_acc
                improved_loss = val_loss < best_val_loss
                
                # Consider both balanced accuracy and loss for early stopping
                if improved_balanced_acc or (improved_loss and epoch < 30):
                    if improved_balanced_acc:
                        best_val_balanced_acc = val_balanced_acc
                    if improved_loss:
                best_val_loss = val_loss
                        
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                best_epoch = epoch
                no_improvement_epochs = 0
                    
                    # Print improvement message
                    if improved_balanced_acc:
                        print(f"  ✓ New best model (balanced acc: {best_val_balanced_acc:.4f})")
                    else:
                print(f"  ✓ New best model (val_loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                no_improvement_epochs += 1
                    print(f"  × No improvement for {no_improvement_epochs} epochs (best balanced acc: {best_val_balanced_acc:.4f}, best val_loss: {best_val_loss:.4f})")
                    
                    # Early stopping with longer patience
                if patience_counter >= early_stopping_patience and epoch > 100:  # Don't stop too early
                    print(f'Early stopping at epoch {epoch+1} (best was epoch {best_epoch+1})')
                    break
                
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user at epoch {}/{}".format(epoch+1, self.n_epochs))
            print("Evaluating best model found so far...")
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Loaded best model from epoch {best_epoch+1}")
        else:
            print("No best model found. Using the most recent model state.")
        
        # Final evaluation
        self.model.eval()
        all_targets = []
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        cm = confusion_matrix(all_targets, all_predictions)
        
        print(f'\nFinal validation accuracy: {accuracy:.4f}')
        print('\nConfusion Matrix:')
        print(cm)
        
        # Print detailed per-class metrics
        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions)
        
        # Calculate balanced accuracy (more meaningful for imbalanced datasets)
        class_accuracies = []
        for i in range(self.n_classes):
            class_correct = np.sum((all_targets == i) & (all_predictions == i))
            class_total = np.sum(all_targets == i)
            if class_total > 0:
                class_acc = class_correct/class_total
                print(f'Class {i} accuracy: {class_acc:.4f} ({class_correct}/{class_total})')
                class_accuracies.append(class_acc)
            else:
                print(f'Class {i} accuracy: N/A (0 samples)')
        
        balanced_accuracy = sum(class_accuracies) / len(class_accuracies)
        print(f'Balanced accuracy: {balanced_accuracy:.4f}')
        
        # Add training history debug information
        history['val_targets_history'] = all_val_targets_history
        history['val_preds_history'] = all_val_preds_history
        
        return {
            'model': self.model,
            'history': history,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': np.array(all_probabilities)
        }
    
    def predict(self, X):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        
        return predicted.cpu().numpy()


def subject_specific_normalize(X, subject_indices=None):
    """
    Apply z-score normalization to each subject's data separately.
    If subject_indices is None, treat all data as coming from a single subject.
    
    Args:
        X: EEG data with shape (n_trials, n_channels, n_times)
        subject_indices: Array of subject indices for each trial, or None
        
    Returns:
        X_normalized: Normalized EEG data with same shape as X
    """
    X_normalized = np.zeros_like(X)
    
    if subject_indices is None:
        # Single subject case - normalize all trials together
        for c in range(X.shape[1]):  # Loop over channels
            # Calculate mean and std across all trials for this channel
            channel_mean = np.mean(X[:, c, :])
            channel_std = np.std(X[:, c, :])
            
            # Avoid division by zero
            if channel_std > 0:
                X_normalized[:, c, :] = (X[:, c, :] - channel_mean) / channel_std
            else:
                X_normalized[:, c, :] = X[:, c, :]
                print(f"Warning: Channel {c} has zero standard deviation")
                
        print(f"Applied subject-specific normalization (single subject): mean={X_normalized.mean():.4f}, std={X_normalized.std():.4f}")
                
    else:
        # Multi-subject case - normalize each subject separately
        unique_subjects = np.unique(subject_indices)
        print(f"Applying subject-specific normalization for {len(unique_subjects)} subjects")
        
        for subject_idx in unique_subjects:
            # Get data for this subject
            subject_mask = subject_indices == subject_idx
            X_subject = X[subject_mask]
            
            # Normalize each channel for this subject
            for c in range(X_subject.shape[1]):
                # Calculate mean and std across all trials for this channel and subject
                channel_mean = np.mean(X_subject[:, c, :])
                channel_std = np.std(X_subject[:, c, :])
                
                # Avoid division by zero
                if channel_std > 0:
                    X_normalized[subject_mask, c, :] = (X_subject[:, c, :] - channel_mean) / channel_std
                else:
                    X_normalized[subject_mask, c, :] = X_subject[:, c, :]
                    print(f"Warning: Subject {subject_idx}, Channel {c} has zero standard deviation")
        
        # Print statistics
        print(f"Applied subject-specific normalization (multi-subject): mean={X_normalized.mean():.4f}, std={X_normalized.std():.4f}")
    
    return X_normalized


def train_freq_time_lstm(subject_id=SUBJECT_ID, run_ids=RUN_IDS, save_model=True):
    """Train a frequency-time domain LSTM model on a single subject's data with subject-specific normalization"""
    # Load and preprocess data
    X, y, info = load_subject_data(subject_id, run_ids)
    
    # Print dataset information
    print(f"\nTraining Frequency-Time LSTM for subject {subject_id}")
    print(f"Data shape: {X.shape} (trials, channels, time points)")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Create classifier wrapper
    classifier = LSTMClassifier(
        n_classes=len(np.unique(y)),
        hidden_size=64,
        num_layers=1,
        dropout=0.5,
        bidirectional=False,
        lr=0.0005,
        batch_size=16,
        n_epochs=300,
        weight_decay=0.01,
        augmentation_factor=3
    )
    
    # Override the _prepare_data method to use our modified version
    classifier._prepare_data = LSTMClassifier._prepare_data.__get__(classifier, LSTMClassifier)
    
    # Replace the model initialization in train_and_evaluate
    n_channels, n_times = X.shape[1], X.shape[2]
    
    # Use the frequency-time model instead
    classifier.model = FrequencyTimeLSTMModel(
        n_channels=n_channels,
        n_classes=classifier.n_classes,
        hidden_size=classifier.hidden_size,
        dropout=classifier.dropout,
        bidirectional=classifier.bidirectional
    ).to(classifier.device)
    
    # Print model summary
    print(f"Frequency-Time LSTM model architecture:\n{classifier.model}")
    total_params = sum(p.numel() for p in classifier.model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Train and evaluate
    results = classifier.train_and_evaluate(X, y, val_split=0.2, early_stopping_patience=50)
    
    # Add the trained classifier to the results
    results['classifier'] = classifier
    
    # Plot training history
    plt.figure(figsize=(15, 12))
    
    # Loss plot
    plt.subplot(3, 2, 1)
    plt.plot(results['history']['train_loss'], 'b-', label='Training Loss')
    plt.plot(results['history']['val_loss'], 'r-', label='Validation Loss')
    plt.title(f'Training and Validation Loss - Subject {subject_id}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(3, 2, 2)
    plt.plot(results['history']['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(results['history']['val_acc'], 'r-', label='Validation Accuracy')
    plt.title(f'Training and Validation Accuracy - Subject {subject_id}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Learning rate plot
    plt.subplot(3, 2, 3)
    plt.plot(results['history']['learning_rates'])
    plt.title('Learning Rate Schedule')
    plt.xlabel('Iterations')
    plt.ylabel('Learning Rate')
    
    # Confusion Matrix
    plt.subplot(3, 2, 4)
    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - Subject {subject_id}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Class prediction distribution over epochs
    plt.subplot(3, 2, 5)
    class_preds = np.array(results['history']['class_preds'])
    for i in range(classifier.n_classes):
        plt.plot(class_preds[:, i], label=f'Class {i}')
    plt.title('Class Prediction Distribution Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Proportion of Predictions')
    plt.legend()
    
    # Final predictions vs. actual
    plt.subplot(3, 2, 6)
    plt.scatter(range(len(results['targets'])), results['targets'], c='blue', alpha=0.5, label='Actual')
    plt.scatter(range(len(results['predictions'])), results['predictions'], c='red', alpha=0.5, label='Predicted')
    plt.title('Validation Set: Actual vs. Predicted')
    plt.xlabel('Sample Index')
    plt.ylabel('Class')
    plt.yticks([0, 1])
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Save the model if requested
    if save_model:
        model_dir = os.path.join(os.getcwd(), 'models')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f'freq_time_lstm_subject_{subject_id}_runs_{"-".join(run_ids)}.pt')
        classifier.save_model(model_path)
    
    # Evaluate on all non-augmented data
    print("\n===== Evaluating on all non-augmented data =====")
    eval_results = classifier.evaluate_on_data(X, y)
    
    print(f"Full dataset accuracy: {eval_results['accuracy']:.4f}")
    
    # Calculate and print balanced accuracy
    class_accuracies = []
    for i in range(classifier.n_classes):
        if i in eval_results['class_accuracies']:
            acc = eval_results['class_accuracies'][i]
            if not np.isnan(acc):
                class_accuracies.append(acc)
                
    balanced_accuracy = sum(class_accuracies) / len(class_accuracies) if class_accuracies else 0
    print(f"Full dataset balanced accuracy: {balanced_accuracy:.4f}")
    
    print("\nFull dataset confusion matrix:")
    print(eval_results['confusion_matrix'])
    
    # Print per-class accuracy
    for i in range(classifier.n_classes):
        acc = eval_results['class_accuracies'][i]
        total = np.sum(eval_results['targets'] == i)
        correct = int(acc * total) if not np.isnan(acc) else 0
        print(f"Class {i} accuracy: {acc:.4f} ({correct}/{total})")
    
    # Plot confusion matrix for full dataset
    plt.figure(figsize=(8, 6))
    sns.heatmap(eval_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - Full Dataset (Subject {subject_id})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    return results


# -----------------------------------------------------------------
# Multi-subject combined model implementation
# -----------------------------------------------------------------

def load_multi_subject_data(subject_ids=None, run_ids=RUN_IDS, use_motor_channels=USE_MOTOR_CHANNELS):
    """
    Load and combine data from multiple subjects, ignoring those with non-standard time lengths
    """
    if subject_ids is None:
        # Use all subjects by default (001-109)
        subject_ids = [f"{i:03d}" for i in range(1, 110)]
    
    # Find the standard time dimension from the first subject
    standard_length = None
    for subject_id in subject_ids:
        try:
            X, _, _ = load_subject_data(subject_id, run_ids, use_motor_channels)
            _, _, standard_length = X.shape
            print(f"Using standard time length of {standard_length} from subject {subject_id}")
            break
        except Exception as e:
            print(f"Error loading subject {subject_id}: {str(e)}")
            continue
    
    if standard_length is None:
        raise ValueError("Could not determine standard time length")
    
    # Process each subject
    all_X = []
    all_y = []
    all_subject_indices = []
    subject_count = 0
    skipped_count = 0
    
    print(f"Loading data from {len(subject_ids)} subjects...")
    
    for subject_id in subject_ids:
        try:
            X, y, _ = load_subject_data(subject_id, run_ids, use_motor_channels)
            
            # Check if time dimension matches standard
            _, _, time_points = X.shape
            if time_points != standard_length:
                print(f"Subject {subject_id}: Skipped - non-standard time dimension ({time_points})")
                skipped_count += 1
                continue
                
            # Track which subject each trial came from
            subject_indices = np.full(len(y), subject_count)
            
            all_X.append(X)
            all_y.append(y)
            all_subject_indices.append(subject_indices)
            
            print(f"Subject {subject_id}: Added {len(y)} trials (classes: {np.bincount(y)})")
            subject_count += 1
            
        except Exception as e:
            print(f"Error loading subject {subject_id}: {str(e)}")
            continue
    
    if not all_X:
        raise ValueError("No valid subject data was loaded")
    
    # Combine data from all subjects
    X_combined = np.concatenate(all_X, axis=0)
    y_combined = np.concatenate(all_y, axis=0)
    subject_indices_combined = np.concatenate(all_subject_indices, axis=0)
    
    print(f"Combined dataset: {X_combined.shape[0]} trials, {X_combined.shape[1]} channels, {X_combined.shape[2]} timepoints")
    print(f"Class distribution: {np.bincount(y_combined)}")
    print(f"Used {subject_count} subjects, skipped {skipped_count} subjects with non-standard time lengths")
    
    return X_combined, y_combined, subject_indices_combined


def train_multi_subject_freq_time(subject_ids=None, run_ids=RUN_IDS, save_model=True):
    """Train a frequency-time domain LSTM model on data combined from multiple subjects with subject-specific normalization"""
    # Load combined data from all subjects
    X, y, subject_indices = load_multi_subject_data(subject_ids, run_ids)
    
    # Get number of unique subjects
    n_subjects = len(np.unique(subject_indices))
    
    # Print dataset information
    print(f"\nTraining Frequency-Time LSTM on combined data from {n_subjects} subjects")
    print(f"Data shape: {X.shape} (trials, channels, time points)")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Apply subject-specific normalization BEFORE splitting into train/val
    print("Applying subject-specific normalization (multi-subject)...")
    X = subject_specific_normalize(X, subject_indices)
    
    # Create classifier for combined data
    classifier = LSTMClassifier(
        n_classes=len(np.unique(y)),
        hidden_size=64,
        num_layers=1,
        dropout=0.5,
        bidirectional=False,
        lr=0.0005,
        batch_size=32,  # Larger batch size for more data
        n_epochs=300,
        weight_decay=0.01,
        augmentation_factor=2  # Lower augmentation factor for more data
    )
    
    # Override the _prepare_data method to use our modified version
    classifier._prepare_data = LSTMClassifier._prepare_data.__get__(classifier, LSTMClassifier)
    
    # Initialize model
    n_channels, n_times = X.shape[1], X.shape[2]
    
    # Use the frequency-time model
    classifier.model = FrequencyTimeLSTMModel(
        n_channels=n_channels,
        n_classes=classifier.n_classes,
        hidden_size=classifier.hidden_size,
        dropout=classifier.dropout,
        bidirectional=classifier.bidirectional
    ).to(classifier.device)
    
    # Print model summary
    print(f"Frequency-Time LSTM model architecture (multi-subject):\n{classifier.model}")
    total_params = sum(p.numel() for p in classifier.model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Train and evaluate using the standard training method
    print("Training on combined data from multiple subjects...")
    results = classifier.train_and_evaluate(X, y, val_split=0.2, early_stopping_patience=50)
    
    # Add the trained classifier to the results
    results['classifier'] = classifier
    
    # Plot training history
    plt.figure(figsize=(15, 12))
    
    # Loss plot
    plt.subplot(3, 2, 1)
    plt.plot(results['history']['train_loss'], 'b-', label='Training Loss')
    plt.plot(results['history']['val_loss'], 'r-', label='Validation Loss')
    plt.title(f'Training and Validation Loss - Multi-Subject')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(3, 2, 2)
    plt.plot(results['history']['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(results['history']['val_acc'], 'r-', label='Validation Accuracy')
    plt.title(f'Training and Validation Accuracy - Multi-Subject')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Learning rate plot
    plt.subplot(3, 2, 3)
    plt.plot(results['history']['learning_rates'])
    plt.title('Learning Rate Schedule')
    plt.xlabel('Iterations')
    plt.ylabel('Learning Rate')
    
    # Confusion Matrix
    plt.subplot(3, 2, 4)
    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - Multi-Subject')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Class prediction distribution over epochs
    plt.subplot(3, 2, 5)
    class_preds = np.array(results['history']['class_preds'])
    for i in range(classifier.n_classes):
        plt.plot(class_preds[:, i], label=f'Class {i}')
    plt.title('Class Prediction Distribution Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Proportion of Predictions')
    plt.legend()
    
    # Final predictions vs. actual
    plt.subplot(3, 2, 6)
    plt.scatter(range(len(results['targets'])), results['targets'], c='blue', alpha=0.5, label='Actual')
    plt.scatter(range(len(results['predictions'])), results['predictions'], c='red', alpha=0.5, label='Predicted')
    plt.title('Validation Set: Actual vs. Predicted')
    plt.xlabel('Sample Index')
    plt.ylabel('Class')
    plt.yticks([0, 1])
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Save the model if requested
    if save_model:
        model_dir = os.path.join(os.getcwd(), 'models')
        os.makedirs(model_dir, exist_ok=True)
        subject_str = 'all' if subject_ids is None else f"{len(subject_ids)}_subjects"
        model_path = os.path.join(model_dir, f'freq_time_lstm_{subject_str}_runs_{"-".join(run_ids)}.pt')
        classifier.save_model(model_path)
    
    # Evaluate on all non-augmented data
    print("\n===== Evaluating on all non-augmented data =====")
    eval_results = classifier.evaluate_on_data(X, y)
    
    print(f"Full dataset accuracy: {eval_results['accuracy']:.4f}")
    
    # Calculate and print balanced accuracy
    class_accuracies = []
    for i in range(classifier.n_classes):
        if i in eval_results['class_accuracies']:
            acc = eval_results['class_accuracies'][i]
            if not np.isnan(acc):
                class_accuracies.append(acc)
                
    balanced_accuracy = sum(class_accuracies) / len(class_accuracies) if class_accuracies else 0
    print(f"Full dataset balanced accuracy: {balanced_accuracy:.4f}")
    
    print("\nFull dataset confusion matrix:")
    print(eval_results['confusion_matrix'])
    
    # Print per-class accuracy
    for i in range(classifier.n_classes):
        acc = eval_results['class_accuracies'][i]
        total = np.sum(eval_results['targets'] == i)
        correct = int(acc * total) if not np.isnan(acc) else 0
        print(f"Class {i} accuracy: {acc:.4f} ({correct}/{total})")
    
    # Plot confusion matrix for full dataset
    plt.figure(figsize=(8, 6))
    sns.heatmap(eval_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Full Dataset (Multiple Subjects)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    return results


if __name__ == "__main__":
    # Uncomment the approach you want to run:
    
    # Single subject with frequency-time features
    # results = train_freq_time_lstm(SUBJECT_ID, RUN_IDS, save_model=True)
    
    # Multi-subject with frequency-time features
    results = train_multi_subject_freq_time(save_model=True)
    
    print(f"\nFinal accuracy: {results['accuracy']:.4f}") 