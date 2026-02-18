"""
Hybrid CNN/Transformer for EEG Classification

This script implements a hybrid CNN/Transformer architecture for EEG classification,
with support for CSP features and frequency domain processing.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from mne.decoding import CSP
import mne
import seaborn as sns
from scipy.signal import welch
import warnings
import time
warnings.filterwarnings('ignore')

# Global configuration (reused from single_subject_lstm.py)
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
    (Reused from single_subject_lstm.py)
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


def extract_frequency_features(X, sfreq=160.0, bands=[(8, 12), (13, 30)]):
    """
    Extract frequency band power features from EEG data
    
    Args:
        X: EEG data with shape (n_trials, n_channels, n_times)
        sfreq: Sampling frequency
        bands: List of frequency bands to extract power for
        
    Returns:
        X_freq: Frequency features with shape (n_trials, n_channels * n_bands)
    """
    n_trials, n_channels, n_times = X.shape
    n_bands = len(bands)
    
    # Initialize output array
    X_freq = np.zeros((n_trials, n_channels * n_bands))
    
    # Calculate power spectrum for each trial and channel
    for trial in range(n_trials):
        for ch in range(n_channels):
            # Calculate power spectral density
            freqs, psd = welch(X[trial, ch, :], fs=sfreq, nperseg=min(256, n_times))
            
            # Extract band power
            for i, band in enumerate(bands):
                # Find frequency range indices
                idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
                
                # Calculate average power in the band
                band_power = np.mean(psd[idx_band])
                
                # Store in output array
                X_freq[trial, ch * n_bands + i] = band_power
    
    # Log-transform the power values to make them more Gaussian-like
    X_freq = np.log(X_freq + 1e-10)  # Adding small constant to avoid log(0)
    
    print(f"Extracted frequency features with shape {X_freq.shape}")
    return X_freq


def extract_csp_features(X, y, n_components=4):
    """
    Extract Common Spatial Pattern features from EEG data
    
    Args:
        X: EEG data with shape (n_trials, n_channels, n_times)
        y: Class labels
        n_components: Number of CSP components to extract
        
    Returns:
        X_csp: CSP features with shape (n_trials, n_components)
    """
    # Initialize CSP
    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    
    # Fit and transform the data
    X_csp = csp.fit_transform(X, y)
    
    print(f"Extracted CSP features with shape {X_csp.shape}")
    return X_csp, csp


def load_multi_subject_data(subject_ids=None, run_ids=RUN_IDS, use_motor_channels=USE_MOTOR_CHANNELS, 
                            max_subjects=None, random_seed=42):
    """
    Load and combine data from multiple subjects, ignoring those with non-standard time lengths
    
    Args:
        subject_ids: Specific subject IDs to load (None = all available)
        run_ids: Run IDs to use
        use_motor_channels: Whether to select only motor-related channels
        max_subjects: Maximum number of subjects to include (None = all)
        random_seed: Random seed for subject sampling
        
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
        np.random.seed(random_seed)
        subject_ids = np.random.choice(subject_ids, size=max_subjects, replace=False)
        print(f"Randomly sampled {max_subjects} subjects: {', '.join(subject_ids)}")
    
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


def subject_specific_normalize(X, subject_indices=None):
    """
    Apply z-score normalization to each subject's data separately.
    If subject_indices is None, treat all data as coming from a single subject.
    (Reused from single_subject_lstm.py)
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


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    """
    Lightweight multi-head self-attention module
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out = nn.Linear(d_model, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # Linear projections and reshape
        q = self.query(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.key(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.value(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            # Handle different mask formats
            if mask.dim() == 2:
                # Convert 1D mask [batch_size, seq_len] to attention mask [batch_size, 1, 1, seq_len]
                mask = mask.unsqueeze(1).unsqueeze(2)
                scores = scores.masked_fill(mask == 0, -1e9)
            elif mask.dim() == 3:
                # Handle 2D mask [batch_size, seq_len, seq_len]
                mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
                scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)
        
        # Reshape and concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.out(context)
        
        return output


class TransformerEncoderLayer(nn.Module):
    """
    Lightweight transformer encoder layer
    """
    def __init__(self, d_model, n_heads, d_ff=1024, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        # Multi-head self-attention
        self.self_attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution for EEG data
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DepthwiseSeparableConv, self).__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            padding='same'
        )
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1
        )
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class EEGCNNEncoder(nn.Module):
    """
    CNN encoder for EEG data inspired by EEGNet
    """
    def __init__(self, n_channels, n_times, embedding_dim=128, dropout=0.2, cnn_internal_shape=None):
        super(EEGCNNEncoder, self).__init__()
        
        # Determine if we need to adapt kernel sizes based on custom internal shape
        self.use_custom_shape = cnn_internal_shape is not None
        self.n_times = n_times
        self.cnn_internal_shape = cnn_internal_shape
        
        # For existing models with hard-coded internal sizes
        conv1_kernel_size = 14 if self.use_custom_shape else 64
        
        # First block: Temporal convolution
        self.block1 = nn.Sequential(
            # Initial temporal convolution
            nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(16),
            # Depthwise convolution
            nn.Conv2d(16, 32, (n_channels, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout)
        )
        
        # Special handling for second block based on custom shapes
        if self.use_custom_shape:
            # Hard-coded for compatibility with existing models
            self.block2 = nn.Sequential(
                # Separable convolution with fixed kernel size for compatibility
                nn.Conv2d(32, 32, (1, 14), padding=(0, 7), groups=32, bias=False),
                nn.Conv2d(32, 32, (1, 1), bias=False),
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.AvgPool2d((1, 8)),
                nn.Dropout(dropout)
            )
        else:
            # Original implementation for new models
            self.block2 = nn.Sequential(
                # Separable convolution
                nn.Conv2d(32, 32, (1, 16), padding=(0, 8), groups=32, bias=False),
                nn.Conv2d(32, 32, (1, 1), bias=False),
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.AvgPool2d((1, 8)),
                nn.Dropout(dropout)
            )
        
        # Calculate output size
        self.n_times_out = cnn_internal_shape if cnn_internal_shape is not None else n_times // 32
        
        # Final convolution to reduce to embedding dimension
        self.final_conv = nn.Conv1d(32, embedding_dim, 1)
        self.flat_size = embedding_dim * self.n_times_out
        
    def forward(self, x):
        # Input shape: [batch_size, n_channels, n_times]
        batch_size = x.size(0)
        
        # Reshape for 2D convolution
        x = x.unsqueeze(1)  # [batch_size, 1, n_channels, n_times]
        
        # Apply first block: Temporal and spatial convolutions
        x = self.block1(x)  # [batch_size, 32, 1, n_times/4]
        
        # Apply second block: Separable convolution
        x = self.block2(x)  # [batch_size, 32, 1, n_times/32]
        
        # Reshape for final convolution
        x = x.squeeze(2)  # [batch_size, 32, n_times/32]
        
        # Apply final convolution
        x = self.final_conv(x)  # [batch_size, embedding_dim, n_times/32]
        
        # Permute to sequence format
        x = x.permute(0, 2, 1)  # [batch_size, n_times/32, embedding_dim]
        
        return x


class HybridCNNTransformer(nn.Module):
    """
    Hybrid CNN/Transformer model for EEG classification
    with CSP and frequency feature integration
    """
    def __init__(self, n_channels, n_times, n_classes, 
                 n_csp_components=4, n_freq_features=None,
                 embedding_dim=128, n_heads=4, n_layers=2, 
                 dropout=0.2, use_csp=True, use_freq=True,
                 cnn_internal_shape=None, seq_len_override=None):
        super(HybridCNNTransformer, self).__init__()
        
        # Store input dimensions as attributes for model serialization
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes
        self.n_csp_components = n_csp_components
        self.n_freq_features = n_freq_features
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.use_csp = use_csp
        self.use_freq = use_freq
        self.cnn_internal_shape = cnn_internal_shape
        
        # CNN encoder for raw EEG
        self.cnn_encoder = EEGCNNEncoder(
            n_channels=n_channels,
            n_times=n_times,
            embedding_dim=embedding_dim,
            dropout=dropout,
            cnn_internal_shape=cnn_internal_shape
        )
        
        # Feature size calculation
        self.n_times_out = cnn_internal_shape if cnn_internal_shape is not None else n_times // 32
        
        # CSP features processing
        if use_csp:
            self.csp_encoder = nn.Sequential(
                nn.Linear(n_csp_components, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        
        # Frequency features processing
        if use_freq and n_freq_features is not None:
            self.freq_encoder = nn.Sequential(
                nn.Linear(n_freq_features, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        
        # Calculate total sequence length for transformer
        self.seq_len = seq_len_override if seq_len_override is not None else self.n_times_out
        if seq_len_override is None:
            if use_csp:
                self.seq_len += 1  # Add one token for CSP features
            if use_freq and n_freq_features is not None:
                self.seq_len += 1  # Add one token for frequency features
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=embedding_dim,
            max_len=self.seq_len,
            dropout=dropout
        )
        
        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=embedding_dim,
                n_heads=n_heads,
                d_ff=embedding_dim * 4,
                dropout=dropout
            ) for _ in range(n_layers)
        ])
        
        # Layer norm before classification
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, n_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
    def forward(self, x_raw, x_csp=None, x_freq=None, attention_mask=None):
        """Forward pass for the hybrid model"""
        # Get batch size
        batch_size = x_raw.size(0)
        
        # Process raw EEG with CNN
        cnn_features = self.cnn_encoder(x_raw)  # [batch_size, n_times_out, embedding_dim]
        
        # Store CNN output shape if not already stored
        if not hasattr(self, 'cnn_output_shape'):
            self.cnn_output_shape = tuple(cnn_features.shape)
        
        # Start building the sequence for transformer
        sequence = [cnn_features]
        
        # Process CSP features if available
        if self.use_csp and x_csp is not None:
            # Encode CSP features to embedding dimension
            csp_features = self.csp_encoder(x_csp)  # [batch_size, embedding_dim]
            
            # Store CSP output shape if not already stored
            if not hasattr(self, 'csp_output_shape'):
                self.csp_output_shape = tuple(csp_features.shape)
            
            # Add as a separate token
            csp_token = csp_features.unsqueeze(1)  # [batch_size, 1, embedding_dim]
            sequence.append(csp_token)
        
        # Process frequency features if available
        if self.use_freq and x_freq is not None:
            # Encode frequency features to embedding dimension
            freq_features = self.freq_encoder(x_freq)  # [batch_size, embedding_dim]
            
            # Store frequency output shape if not already stored
            if not hasattr(self, 'freq_output_shape'):
                self.freq_output_shape = tuple(freq_features.shape)
            
            # Add as a separate token
            freq_token = freq_features.unsqueeze(1)  # [batch_size, 1, embedding_dim]
            sequence.append(freq_token)
        
        # Concatenate all features into a single sequence
        sequence = torch.cat(sequence, dim=1)  # [batch_size, seq_len, embedding_dim]
        
        # Apply positional encoding
        sequence = self.pos_encoder(sequence)
        
        # Handle variable-length sequences if attention mask is provided
        if attention_mask is not None:
            # Adjust mask for sequence length after features concatenation
            seq_len = sequence.size(1)
            if attention_mask.size(1) > seq_len:
                attention_mask = attention_mask[:, :seq_len]
            elif attention_mask.size(1) < seq_len:
                # Pad with ones (attend to all new tokens)
                padding = torch.ones(batch_size, seq_len - attention_mask.size(1), device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, padding], dim=1)
                
            # Create 2D attention mask
            attn_mask = attention_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
        else:
            # Create a full attention mask (all positions attend to all others)
            seq_len = sequence.size(1)
            attn_mask = torch.ones(batch_size, seq_len, seq_len, device=sequence.device)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            sequence = layer(sequence, attn_mask)
        
        # Global average pooling (considering mask)
        if attention_mask is not None:
            # Expanded mask for pooling with variable-length sequences
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(sequence)
            sequence = sequence * mask_expanded
            pooled = sequence.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            # Standard pooling for fixed-length sequences
            pooled = torch.mean(sequence, dim=1)
        
        # Apply layer normalization
        normalized = self.layer_norm(pooled)
        
        # Apply classification head
        logits = self.classifier(normalized)
        
        return logits


class EEGDataAugmenter:
    """
    Class for applying data augmentation techniques to EEG data
    (Modified from single_subject_lstm.py)
    """
    def __init__(self, noise_level=0.03, scale_range=(0.9, 1.1),
                 chunk_max_offset=30, support_variable_length=True):
        """
        Initialize the data augmenter with various augmentation parameters
        
        Args:
            noise_level: Standard deviation of Gaussian noise to add
            scale_range: Range for random amplitude scaling
            chunk_max_offset: Maximum offset for chunk selection
            support_variable_length: Whether to support variable-length outputs
        """
        self.noise_level = noise_level
        self.scale_range = scale_range
        self.chunk_max_offset = chunk_max_offset
        self.support_variable_length = support_variable_length
        # Store a cache of trials for mixup, will be populated during augmentation
        self.trial_cache = None
        self.trial_labels = None
    
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
        
        If support_variable_length is True, returns a shorter sequence
        Otherwise, pads with zeros to maintain original length
        """
        n_channels, n_times = x.shape
        
        # Determine maximum offset based on signal length
        max_offset = min(self.chunk_max_offset, n_times // 6)
        
        # Generate random offset (start point)
        offset = np.random.randint(0, max_offset + 1)
        
        if offset == 0:
            # No offset, return original
            return x.copy()
            
        if self.support_variable_length:
            # Return shorter sequence without zero-padding
            return x[:, offset:].copy()
        else:
            # Legacy mode: Create a new array with the same shape
            x_new = np.zeros_like(x)
            
            # Fill with the selected chunk, padding with zeros
            x_new[:, :-offset] = x[:, offset:]
            
            return x_new
    
    def variable_length_sequence(self, x):
        """
        Create a variable length sequence by randomly trimming the end
        and/or beginning of the signal with tapering to avoid abrupt changes
        
        If support_variable_length is True, returns a shorter sequence
        Otherwise, pads with zeros to maintain original length
        """
        n_channels, n_times = x.shape
        
        # Don't trim more than 15% from each end
        max_trim = int(n_times * 0.15)
        
        # Random amount to trim from start and end
        start_trim = np.random.randint(0, max_trim)
        end_trim = np.random.randint(0, max_trim)
        
        # Extract trimmed sequence
        trimmed_seq = x[:, start_trim:n_times-end_trim]
        
        # Apply tapering to avoid abrupt transitions
        taper_length = min(20, trimmed_seq.shape[1] // 4)
        
        # Create a copy to avoid modifying the original
        tapered_seq = trimmed_seq.copy()
        
        if taper_length > 0:
            taper_in = np.linspace(0, 1, taper_length)
            taper_out = np.linspace(1, 0, taper_length)
            
            for c in range(n_channels):
                # Apply tapering to beginning and end of the trimmed sequence
                tapered_seq[c, :taper_length] *= taper_in
                tapered_seq[c, -taper_length:] *= taper_out
        
        if self.support_variable_length:
            # Return the variable-length sequence directly
            return tapered_seq
        else:
            # Legacy mode: Place in center of output array with zero padding
            x_new = np.zeros_like(x)
            trim_length = tapered_seq.shape[1]
            start_idx = (n_times - trim_length) // 2
            x_new[:, start_idx:start_idx+trim_length] = tapered_seq
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
    
    # New specialized EEG augmentation methods
    
    def spectral_perturbation(self, x):
        """
        Perturb specific frequency bands without time-domain distortion
        
        This augmentation preserves the overall spectral profile while introducing
        controlled variability in specific frequency bands relevant to EEG analysis.
        Unlike frequency_band_noise, this perturbs the entire spectrum with different
        intensities per band.
        """
        n_channels, n_times = x.shape
        x_new = x.copy()
        fs = 160  # Sampling rate
        
        # Define frequency bands of interest for motor imagery
        bands = [
            (0, 4, 0.05),    # Delta (low perturbation)
            (4, 8, 0.08),    # Theta (moderate perturbation)
            (8, 12, 0.15),   # Mu (high perturbation - important for motor imagery)
            (13, 30, 0.12),  # Beta (high perturbation - important for motor imagery)
            (30, 50, 0.05)   # Gamma (low perturbation)
        ]
        
        for c in range(n_channels):
            # Convert to frequency domain
            x_fft = np.fft.rfft(x[c])
            freqs = np.fft.rfftfreq(n_times, d=1/fs)
            
            # Apply different perturbation to each band
            for band_min, band_max, perturb_factor in bands:
                # Create mask for this band
                band_mask = (freqs >= band_min) & (freqs <= band_max)
                
                if np.any(band_mask):
                    # Generate multiplicative perturbation that preserves relative power
                    # Using log-normal distribution to ensure positive values
                    perturbation = np.exp(np.random.normal(0, perturb_factor, size=sum(band_mask)))
                    
                    # Apply perturbation while preserving phase
                    magnitude = np.abs(x_fft[band_mask])
                    phase = np.angle(x_fft[band_mask])
                    
                    # Apply perturbation to magnitude, preserve phase
                    new_magnitude = magnitude * perturbation
                    x_fft[band_mask] = new_magnitude * np.exp(1j * phase)
            
            # Convert back to time domain
            x_new[c] = np.fft.irfft(x_fft, n=n_times)
        
        return x_new
    
    def smooth_warping(self, x):
        """
        Apply smooth, physiologically plausible warping to EEG signals
        
        This is a simplified version of Gaussian Process warping that creates
        a smooth warping field to introduce temporal variability while preserving
        the signal's overall structure and frequency characteristics.
        """
        n_channels, n_times = x.shape
        x_warped = np.zeros_like(x)
        
        # Create a smooth warping function with controlled variability
        # Using a sum of sine waves at different frequencies for smoothness
        t = np.linspace(0, 2*np.pi, n_times)
        n_components = np.random.randint(3, 6)  # Use 3-5 frequency components
        warping = np.zeros(n_times)
        
        for i in range(n_components):
            # Random frequency, phase and amplitude for each component
            freq = np.random.uniform(1, 3)
            phase = np.random.uniform(0, 2*np.pi)
            amp = np.random.uniform(0.02, 0.06) / np.sqrt(n_components)
            warping += amp * np.sin(freq * t + phase)
        
        # Convert warping to indices (forward warping)
        indices = np.arange(n_times)
        warped_indices = indices * (1 + warping)  # Stretch/compress the time axis
        
        # Ensure indices are within bounds
        warped_indices = np.clip(warped_indices, 0, n_times - 1)
        
        # Apply warping to each channel separately
        for c in range(n_channels):
            # Use spline interpolation for smoother results
            x_warped[c] = np.interp(indices, warped_indices, x[c])
        
        return x_warped
    
    def trial_mixup(self, x, y=None, same_class_only=True, alpha=0.2):
        """
        Create new trials by mixing existing ones
        
        This method creates a new trial by linearly interpolating between the current
        trial and another randomly selected trial. If same_class_only is True, it will
        only mix with trials of the same class.
        
        Args:
            x: The current trial to augment
            y: The label of the current trial (required if same_class_only=True)
            same_class_only: Whether to only mix with trials of the same class
            alpha: Parameter for the Beta distribution controlling mixing ratio
        """
        # If we don't have cached trials or need to enforce class matching but don't have labels
        if self.trial_cache is None or (same_class_only and (y is None or self.trial_labels is None)):
            # Cannot perform mixup without cached trials
            return x.copy()
        
        # Select a trial to mix with
        if same_class_only and y is not None:
            # Find trials of the same class
            same_class_indices = np.where(self.trial_labels == y)[0]
            if len(same_class_indices) > 0:
                mix_idx = np.random.choice(same_class_indices)
            else:
                # If no trials of the same class, return original
                return x.copy()
        else:
            # Select any random trial
            mix_idx = np.random.randint(0, len(self.trial_cache))
        
        # Get the trial to mix with
        x2 = self.trial_cache[mix_idx]
        
        # If shapes don't match, return original
        if x.shape != x2.shape:
            return x.copy()
        
        # Generate mixing ratio from Beta distribution
        mix_ratio = np.random.beta(alpha, alpha)
        
        # Mix the trials
        return mix_ratio * x + (1 - mix_ratio) * x2
    
    def augment(self, x, y=None):
        """Apply multiple augmentation techniques to a single trial"""
        # Updated list of augmentations (removed phase_randomization and mild_temporal_warp)
        augmentations = [
            self.add_noise,
            self.scale_amplitude,
            self.chunk_selection,
            self.variable_length_sequence,
            self.frequency_band_noise,
            self.spectral_perturbation,
            self.smooth_warping
            # trial_mixup needs special handling as it requires additional parameters
        ]
        
        # Apply 2-3 random augmentations
        num_augmentations = np.random.randint(2, 4)
        selected_augmentations = np.random.choice(augmentations, 
                                                    size=min(num_augmentations, len(augmentations)), 
                                                    replace=False)
        
        # Apply selected augmentations
        x_aug = x.copy()
        for augmentation in selected_augmentations:
            x_aug = augmentation(x_aug)
        
        # Separately decide whether to apply mixup (with 30% probability)
        if self.trial_cache is not None and np.random.rand() < 0.3:
            x_aug = self.trial_mixup(x_aug, y, same_class_only=True)
            
        return x_aug


class HybridModelClassifier:
    """
    Wrapper class for training and evaluating the Hybrid CNN/Transformer model
    """
    def __init__(self, n_classes, embedding_dim=128, n_heads=4, n_layers=2,
                 dropout=0.3, lr=0.0005, batch_size=32, n_epochs=200,
                 weight_decay=0.01, device=None, augmentation_factor=2,
                 use_csp=True, use_freq=True, n_csp_components=4,
                 variable_length_augmentation=True):
        
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.weight_decay = weight_decay
        self.augmentation_factor = augmentation_factor
        self.use_csp = use_csp
        self.use_freq = use_freq
        self.n_csp_components = n_csp_components
        self.variable_length_augmentation = variable_length_augmentation
        
        # Device setup
        self.device = device if device is not None else DEVICE
        print(f"Using device: {self.device}")
        
        # Data augmenter
        self.augmenter = EEGDataAugmenter(support_variable_length=variable_length_augmentation)
        
        # Model will be initialized when data dimensions are known
        self.model = None
        self.csp_transformer = None  # Will store the fitted CSP transformer
    
    def save_model(self, filepath, metadata=None):
        """
        Save the model and its parameters to a file.
        
        Args:
            filepath: Path to save the model
            metadata: Optional dictionary with additional metadata to save
        """
        if metadata is None:
            metadata = {}
            
        # Extract the actual input dimensions from the model
        input_dims = {
            'n_channels': self.model.n_channels,
            'n_times': self.model.n_times,
            'n_freq_features': self.model.n_freq_features,
            'cnn_output_shape': self.model.cnn_output_shape if hasattr(self.model, 'cnn_output_shape') else None,
            'freq_output_shape': self.model.freq_output_shape if hasattr(self.model, 'freq_output_shape') else None,
            'csp_output_shape': self.model.csp_output_shape if hasattr(self.model, 'csp_output_shape') else None,
            'seq_len': self.model.seq_len if hasattr(self.model, 'seq_len') else None
        }
        
        # Save the entire configuration and model
        save_dict = {
            'model_config': {
                'n_classes': self.n_classes,
                'embedding_dim': self.embedding_dim,
                'n_heads': self.n_heads,
                'n_layers': self.n_layers,
                'dropout': self.dropout,
                'use_csp': self.use_csp,
                'use_freq': self.use_freq,
                'n_csp_components': self.n_csp_components
            },
            'input_dims': input_dims,
            'train_config': {
                'lr': self.lr,
                'batch_size': self.batch_size,
                'n_epochs': self.n_epochs,
                'weight_decay': self.weight_decay,
                'augmentation_factor': self.augmentation_factor,
                'variable_length_augmentation': self.variable_length_augmentation
            },
            'metadata': metadata
        }
        
        # Save configuration to a separate file for reference
        config_path = filepath.replace('.pt', '_config.pt')
        torch.save(save_dict, config_path)
        print(f"Saved model configuration to {config_path}")
        
        # Save the full model (includes architecture and parameters)
        torch.save(self.model, filepath)
        print(f"Saved full model to {filepath}")
        
        # Also create a TorchScript version for maximum portability
        try:
            # Create a directory for the model if it doesn't exist
            model_dir = os.path.dirname(filepath)
            if model_dir and not os.path.exists(model_dir):
                os.makedirs(model_dir)
                
            scripted_path = filepath.replace('.pt', '_scripted.pt')
            
            # Ensure model is in eval mode for tracing
            self.model.eval()
            
            # Try tracing instead of scripting as it's more tolerant
            # Create a sample input of appropriate shape
            n_channels = self.model.cnn_encoder.n_channels
            n_times = self.model.cnn_encoder.n_times
            
            # Create dummy inputs that match expected shapes
            dummy_input = torch.randn(1, n_channels, n_times, device=self.device)
            dummy_csp = torch.randn(1, self.model.n_csp_components * 2, device=self.device) if self.use_csp else None
            dummy_freq = torch.randn(1, self.model.n_freq_features, device=self.device) if self.use_freq else None
            
            # Use tracing instead of scripting
            traced_model = torch.jit.trace(self.model, (dummy_input, dummy_csp, dummy_freq))
            traced_model.save(scripted_path)
            print(f"Saved traced TorchScript model to {scripted_path}")
        except Exception as e:
            print(f"Warning: Could not save TorchScript model: {str(e)}")
            print("This is non-critical; the regular model was still saved.")
    
    @classmethod
    def load_model(cls, filepath, device=None):
        """
        Load a saved model.
        
        Args:
            filepath: Path to the saved model
            device: Device to load the model to
            
        Returns:
            Loaded model instance and metadata
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading model from {filepath}")
        
        try:
            # Load the full model - explicitly set weights_only=False for PyTorch 2.6+ compatibility
            model = torch.load(filepath, map_location=device, weights_only=False)
            
            if not isinstance(model, torch.nn.Module):
                raise ValueError("File does not contain a full model. This method only supports loading full models.")
            
            print("Loaded full model architecture and weights")
            
            # Try to load config file for classifier parameters
            config_path = filepath.replace('.pt', '_config.pt')
            if os.path.exists(config_path):
                config_dict = torch.load(config_path, map_location=device, weights_only=False)
                model_config = config_dict.get('model_config', {})
                train_config = config_dict.get('train_config', {})
                metadata = config_dict.get('metadata', {})
            else:
                # Create default config and empty metadata
                print("Warning: Config file not found. Using default classifier parameters.")
                model_config = {}
                train_config = {}
                metadata = {}
            
            # Create a classifier instance
            classifier = cls(
                n_classes=getattr(model, 'n_classes', 4),
                embedding_dim=getattr(model, 'embedding_dim', 64),
                n_heads=getattr(model, 'n_heads', 4),
                n_layers=getattr(model, 'n_layers', 2),
                dropout=getattr(model, 'dropout', 0.2),
                lr=train_config.get('lr', 0.0005),
                batch_size=train_config.get('batch_size', 32),
                n_epochs=train_config.get('n_epochs', 150),
                weight_decay=train_config.get('weight_decay', 0.01),
                device=device,
                augmentation_factor=train_config.get('augmentation_factor', 2),
                use_csp=getattr(model, 'use_csp', True),
                use_freq=getattr(model, 'use_freq', True),
                n_csp_components=getattr(model, 'n_csp_components', 4),
                variable_length_augmentation=train_config.get('variable_length_augmentation', False)
            )
            
            # Set the model
            classifier.model = model.to(device)
            
            print(f"Successfully loaded model from {filepath}")
            return classifier, metadata
                
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise e
    
    def prepare_data(self, X_raw, y, X_csp=None, X_freq=None, val_split=0.2, apply_augmentation=True, 
                     train_indices=None, val_indices=None):
        """
        Prepare data for training with optional augmentation
        """
        # Print original class distribution
        print(f"Original class distribution: {np.bincount(y)}")
        
        # Use provided indices if available, otherwise create stratified split
        if train_indices is None or val_indices is None:
            indices_by_class = [np.where(y == i)[0] for i in range(self.n_classes)]
            
            train_indices = []
            val_indices = []
            
            for class_indices in indices_by_class:
                np.random.shuffle(class_indices)
                val_size = int(len(class_indices) * val_split)
                
                val_indices.extend(class_indices[:val_size])
                train_indices.extend(class_indices[val_size:])
        
        # Split the raw data
        X_raw_train = X_raw[train_indices]
        y_train = y[train_indices]
        X_raw_val = X_raw[val_indices]
        y_val = y[val_indices]
        
        # Split CSP features if provided
        X_csp_train = X_csp[train_indices] if X_csp is not None else None
        X_csp_val = X_csp[val_indices] if X_csp is not None else None
        
        # Split frequency features if provided
        X_freq_train = X_freq[train_indices] if X_freq is not None else None
        X_freq_val = X_freq[val_indices] if X_freq is not None else None
        
        # Apply data augmentation if requested
        if apply_augmentation:
            # Only augment raw EEG data, not the extracted features
            if self.variable_length_augmentation:
                # Variable-length augmentation
                X_raw_train_aug, y_train_aug = self.augmenter.augment_batch(
                    X_raw_train, y_train, self.augmentation_factor
                )
                
                # X_raw_train_aug is now a list of augmented trials with potentially different lengths
                print(f"Created {len(X_raw_train_aug)} augmented trials (includes original {len(X_raw_train)} trials)")
                print(f"Augmented data shapes vary: example shape: {X_raw_train_aug[0].shape}")
            else:
                # Legacy fixed-length augmentation
                X_raw_train_aug, y_train_aug = self.augmenter.augment_batch(
                    X_raw_train, y_train, self.augmentation_factor
                )
                
                # Create augmented versions of feature data
                if X_csp_train is not None:
                    # Duplicate CSP features for each augmented trial
                    X_csp_train_expanded = np.repeat(X_csp_train, self.augmentation_factor + 1, axis=0)
                    X_csp_train = X_csp_train_expanded
                
                if X_freq_train is not None:
                    # Duplicate frequency features for each augmented trial
                    X_freq_train_expanded = np.repeat(X_freq_train, self.augmentation_factor + 1, axis=0)
                    X_freq_train = X_freq_train_expanded
                
                # Update training data with augmented data
                X_raw_train = X_raw_train_aug
                y_train = y_train_aug
                
                print(f"Applied augmentation: {len(train_indices)} trials → {X_raw_train.shape[0]} trials")
                print(f"Augmented class distribution: {np.bincount(y_train)}")
            
        # Normalize data (with different handling for variable-length)
        if self.variable_length_augmentation and apply_augmentation:
            # Normalize each sequence individually
            for i in range(len(X_raw_train_aug)):
                X_raw_train_aug[i] = self._normalize_single_sequence(X_raw_train_aug[i])
            
            # Normalize validation data
            X_raw_val = subject_specific_normalize(X_raw_val)
        else:
            # Standard normalization
            X_raw_train = subject_specific_normalize(X_raw_train)
            X_raw_val = subject_specific_normalize(X_raw_val)
        
        # Normalize CSP and frequency features if provided
        if X_csp_train is not None and X_csp_val is not None:
            # Standardize CSP features
            csp_mean = np.mean(X_csp_train, axis=0)
            csp_std = np.std(X_csp_train, axis=0)
            X_csp_train = (X_csp_train - csp_mean) / (csp_std + 1e-8)
            X_csp_val = (X_csp_val - csp_mean) / (csp_std + 1e-8)
        
        if X_freq_train is not None and X_freq_val is not None:
            # Standardize frequency features
            freq_mean = np.mean(X_freq_train, axis=0)
            freq_std = np.std(X_freq_train, axis=0)
            X_freq_train = (X_freq_train - freq_mean) / (freq_std + 1e-8)
            X_freq_val = (X_freq_val - freq_mean) / (freq_std + 1e-8)
        
        # Convert to PyTorch tensors with different handling for variable-length
        if self.variable_length_augmentation and apply_augmentation:
            train_dataset = self._create_variable_length_dataset(
                X_raw_train_aug, y_train_aug, X_csp_train, X_freq_train
            )
            
            # Validation set is fixed length
            X_raw_val_tensor = torch.FloatTensor(X_raw_val)
            y_val_tensor = torch.LongTensor(y_val)
            
            if X_csp_val is not None:
                X_csp_val_tensor = torch.FloatTensor(X_csp_val)
            else:
                X_csp_val_tensor = None
                
            if X_freq_val is not None:
                X_freq_val_tensor = torch.FloatTensor(X_freq_val)
            else:
                X_freq_val_tensor = None
            
            # Create validation dataset
            if X_csp_val_tensor is not None and X_freq_val_tensor is not None:
                val_dataset = TensorDataset(X_raw_val_tensor, X_csp_val_tensor, X_freq_val_tensor, y_val_tensor)
            elif X_csp_val_tensor is not None:
                val_dataset = TensorDataset(X_raw_val_tensor, X_csp_val_tensor, y_val_tensor)
            elif X_freq_val_tensor is not None:
                val_dataset = TensorDataset(X_raw_val_tensor, X_freq_val_tensor, y_val_tensor)
            else:
                val_dataset = TensorDataset(X_raw_val_tensor, y_val_tensor)
        else:
            # Standard fixed-length tensors
            X_raw_train_tensor = torch.FloatTensor(X_raw_train)
            X_raw_val_tensor = torch.FloatTensor(X_raw_val)
            y_train_tensor = torch.LongTensor(y_train)
            y_val_tensor = torch.LongTensor(y_val)
            
            # Prepare CSP tensors if available
            X_csp_train_tensor = torch.FloatTensor(X_csp_train) if X_csp_train is not None else None
            X_csp_val_tensor = torch.FloatTensor(X_csp_val) if X_csp_val is not None else None
            
            # Prepare frequency tensors if available
            X_freq_train_tensor = torch.FloatTensor(X_freq_train) if X_freq_train is not None else None
            X_freq_val_tensor = torch.FloatTensor(X_freq_val) if X_freq_val is not None else None
            
            # Create datasets
            if X_csp_train_tensor is not None and X_freq_train_tensor is not None:
                # Dataset with all features
                train_dataset = TensorDataset(
                    X_raw_train_tensor, X_csp_train_tensor, 
                    X_freq_train_tensor, y_train_tensor
                )
                val_dataset = TensorDataset(
                    X_raw_val_tensor, X_csp_val_tensor,
                    X_freq_val_tensor, y_val_tensor
                )
            elif X_csp_train_tensor is not None:
                # Dataset with raw and CSP features
                train_dataset = TensorDataset(
                    X_raw_train_tensor, X_csp_train_tensor, y_train_tensor
                )
                val_dataset = TensorDataset(
                    X_raw_val_tensor, X_csp_val_tensor, y_val_tensor
                )
            elif X_freq_train_tensor is not None:
                # Dataset with raw and frequency features
                train_dataset = TensorDataset(
                    X_raw_train_tensor, X_freq_train_tensor, y_train_tensor
                )
                val_dataset = TensorDataset(
                    X_raw_val_tensor, X_freq_val_tensor, y_val_tensor
                )
            else:
                # Dataset with only raw features
                train_dataset = TensorDataset(X_raw_train_tensor, y_train_tensor)
                val_dataset = TensorDataset(X_raw_val_tensor, y_val_tensor)
        
        # Create dataloaders with custom collation for variable-length
        if self.variable_length_augmentation and apply_augmentation:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=self._variable_length_collate_fn
            )
        else:
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
        
        if self.variable_length_augmentation and apply_augmentation:
            print(f"Created data loaders: {len(train_indices)} original training samples, "
                  f"{len(X_raw_train_aug)} after augmentation, {len(val_indices)} validation samples")
        else:
            print(f"Created data loaders: {len(train_indices)} original training samples, "
                  f"{X_raw_train.shape[0]} after augmentation, {len(val_indices)} validation samples")
        
        return train_loader, val_loader
    
    def _normalize_single_sequence(self, x):
        """Normalize a single EEG sequence by channel"""
        x_normalized = np.zeros_like(x)
        
        for c in range(x.shape[0]):  # Loop over channels
            # Calculate mean and std for this channel
            channel_mean = np.mean(x[c, :])
            channel_std = np.std(x[c, :])
            
            # Avoid division by zero
            if channel_std > 0:
                x_normalized[c, :] = (x[c, :] - channel_mean) / channel_std
            else:
                x_normalized[c, :] = x[c, :]
                
        return x_normalized
    
    def _create_variable_length_dataset(self, X_raw_list, y, X_csp=None, X_freq=None):
        """Create a dataset that handles variable-length EEG signals"""
        dataset = []
        
        for i, x_raw in enumerate(X_raw_list):
            # Get label (assuming y is already repeated for each augmented trial)
            label = y[i]
            
            # Convert to tensor
            x_raw_tensor = torch.FloatTensor(x_raw)
            
            # Get corresponding CSP and frequency features if available
            if X_csp is not None:
                x_csp_tensor = torch.FloatTensor(X_csp[i])
            else:
                x_csp_tensor = None
                
            if X_freq is not None:
                x_freq_tensor = torch.FloatTensor(X_freq[i])
            else:
                x_freq_tensor = None
            
            # Store all together
            if x_csp_tensor is not None and x_freq_tensor is not None:
                dataset.append((x_raw_tensor, x_csp_tensor, x_freq_tensor, torch.LongTensor([label])))
            elif x_csp_tensor is not None:
                dataset.append((x_raw_tensor, x_csp_tensor, torch.LongTensor([label])))
            elif x_freq_tensor is not None:
                dataset.append((x_raw_tensor, x_freq_tensor, torch.LongTensor([label])))
            else:
                dataset.append((x_raw_tensor, torch.LongTensor([label])))
        
        return dataset
    
    def _variable_length_collate_fn(self, batch):
        """Custom collate function for variable-length EEG sequences"""
        # Determine number of elements in each sample
        n_elements = len(batch[0])
        
        # Handle different cases based on available features
        if n_elements == 4:  # Raw, CSP, Freq, Labels
            # Extract elements
            x_raw = [item[0] for item in batch]
            x_csp = torch.stack([item[1] for item in batch])
            x_freq = torch.stack([item[2] for item in batch])
            y = torch.cat([item[3] for item in batch])
            
            # Create attention mask for raw signals
            max_len = max([x.shape[1] for x in x_raw])
            attention_masks = []
            padded_x_raw = []
            
            for x in x_raw:
                # Create mask (1 for real signal, 0 for padding)
                seq_len = x.shape[1]
                mask = torch.ones(max_len)
                mask[seq_len:] = 0
                attention_masks.append(mask)
                
                # Pad signal to max_len
                padded = torch.zeros(x.shape[0], max_len)
                padded[:, :seq_len] = x
                padded_x_raw.append(padded)
            
            # Stack tensors
            x_raw_padded = torch.stack(padded_x_raw)
            attention_masks = torch.stack(attention_masks)
            
            return x_raw_padded, attention_masks, x_csp, x_freq, y
            
        elif n_elements == 3:  # Could be (Raw, CSP, Labels) or (Raw, Freq, Labels)
            # Check if second element is CSP or Freq (based on shape)
            if batch[0][1].ndim == 1:  # CSP features are 1D
                # Extract elements
                x_raw = [item[0] for item in batch]
                x_csp = torch.stack([item[1] for item in batch])
                y = torch.cat([item[2] for item in batch])
                
                # Create attention mask for raw signals
                max_len = max([x.shape[1] for x in x_raw])
                attention_masks = []
                padded_x_raw = []
                
                for x in x_raw:
                    # Create mask (1 for real signal, 0 for padding)
                    seq_len = x.shape[1]
                    mask = torch.ones(max_len)
                    mask[seq_len:] = 0
                    attention_masks.append(mask)
                    
                    # Pad signal to max_len
                    padded = torch.zeros(x.shape[0], max_len)
                    padded[:, :seq_len] = x
                    padded_x_raw.append(padded)
                
                # Stack tensors
                x_raw_padded = torch.stack(padded_x_raw)
                attention_masks = torch.stack(attention_masks)
                
                return x_raw_padded, attention_masks, x_csp, y
            else:
                # Extract elements
                x_raw = [item[0] for item in batch]
                x_freq = torch.stack([item[1] for item in batch])
                y = torch.cat([item[2] for item in batch])
                
                # Create attention mask for raw signals
                max_len = max([x.shape[1] for x in x_raw])
                attention_masks = []
                padded_x_raw = []
                
                for x in x_raw:
                    # Create mask (1 for real signal, 0 for padding)
                    seq_len = x.shape[1]
                    mask = torch.ones(max_len)
                    mask[seq_len:] = 0
                    attention_masks.append(mask)
                    
                    # Pad signal to max_len
                    padded = torch.zeros(x.shape[0], max_len)
                    padded[:, :seq_len] = x
                    padded_x_raw.append(padded)
                
                # Stack tensors
                x_raw_padded = torch.stack(padded_x_raw)
                attention_masks = torch.stack(attention_masks)
                
                return x_raw_padded, attention_masks, x_freq, y
        else:
            # Extract elements (Raw, Labels only)
            x_raw = [item[0] for item in batch]
            y = torch.cat([item[1] for item in batch])
            
            # Create attention mask for raw signals
            max_len = max([x.shape[1] for x in x_raw])
            attention_masks = []
            padded_x_raw = []
            
            for x in x_raw:
                # Create mask (1 for real signal, 0 for padding)
                seq_len = x.shape[1]
                mask = torch.ones(max_len)
                mask[seq_len:] = 0
                attention_masks.append(mask)
                
                # Pad signal to max_len
                padded = torch.zeros(x.shape[0], max_len)
                padded[:, :seq_len] = x
                padded_x_raw.append(padded)
            
            # Stack tensors
            x_raw_padded = torch.stack(padded_x_raw)
            attention_masks = torch.stack(attention_masks)
            
            return x_raw_padded, attention_masks, y

    def train_and_evaluate(self, X_raw, y, val_split=0.2, early_stopping_patience=30):
        """
        Train and evaluate the hybrid model with early stopping
        """
        # Get data dimensions
        n_trials, n_channels, n_times = X_raw.shape
        
        # Extract CSP features if requested
        X_csp = None
        if self.use_csp:
            print("Extracting CSP features...")
            X_csp, self.csp_transformer = extract_csp_features(
                X_raw, y, n_components=self.n_csp_components
            )
        
        # Extract frequency features if requested
        X_freq = None
        if self.use_freq:
            print("Extracting frequency features...")
            X_freq = extract_frequency_features(
                X_raw, bands=[(8, 12), (13, 30)]  # Mu and beta bands
            )
            n_freq_features = X_freq.shape[1]
        else:
            n_freq_features = None
        
        # Initialize model
        self.model = HybridCNNTransformer(
            n_channels=n_channels,
            n_times=n_times,
            n_classes=self.n_classes,
            n_csp_components=self.n_csp_components,
            n_freq_features=n_freq_features,
            embedding_dim=self.embedding_dim,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
            use_csp=self.use_csp,
            use_freq=self.use_freq
        ).to(self.device)
        
        # Print model summary
        print(f"Hybrid CNN/Transformer model architecture:\n{self.model}")
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params:,}")
        
        # Create stratified split for training and validation
        indices_by_class = [np.where(y == i)[0] for i in range(self.n_classes)]
        
        train_indices = []
        val_indices = []
        
        for class_indices in indices_by_class:
            np.random.shuffle(class_indices)
            val_size = int(len(class_indices) * val_split)
            
            val_indices.extend(class_indices[:val_size])
            train_indices.extend(class_indices[val_size:])
        
        # Store val_indices for later use
        val_indices = np.array(val_indices)
        
        # Prepare data
        train_loader, val_loader = self.prepare_data(
            X_raw, y, X_csp, X_freq, val_split, apply_augmentation=True,
            train_indices=train_indices, val_indices=val_indices
        )
        
        # Set up optimizer and loss function
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        # Class weights for imbalanced data
        class_counts = np.bincount(y)
        class_weights = torch.FloatTensor(
            np.sqrt(np.sum(class_counts) / (len(class_counts) * class_counts))
        ).to(self.device)
        print(f"Class weights: {class_weights}")
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_balanced_acc': []
        }
        
        # Early stopping variables
        best_val_loss = float('inf')
        best_balanced_acc = 0
        patience_counter = 0
        best_model_state = None
        
        print("\nTraining started. Press Ctrl+C to manually stop training and save the best model so far.")
        
        try:
            # Training loop
            for epoch in range(self.n_epochs):
                # Training phase
                self.model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for batch in train_loader:
                    # Unpack batch
                    if len(batch) == 4:  # Raw, CSP, Freq, Labels
                        X_raw_batch, X_csp_batch, X_freq_batch, y_batch = [b.to(self.device) for b in batch]
                    elif len(batch) == 3:
                        if self.use_csp and not self.use_freq:  # Raw, CSP, Labels
                            X_raw_batch, X_csp_batch, y_batch = [b.to(self.device) for b in batch]
                            X_freq_batch = None
                        else:  # Raw, Freq, Labels
                            X_raw_batch, X_freq_batch, y_batch = [b.to(self.device) for b in batch]
                            X_csp_batch = None
                    else:  # Raw, Labels
                        X_raw_batch, y_batch = [b.to(self.device) for b in batch]
                        X_csp_batch = None
                        X_freq_batch = None
                    
                    # Zero gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = self.model(X_raw_batch, X_csp_batch, X_freq_batch)
                    loss = criterion(outputs, y_batch)
                    
                    # Backward pass and optimization
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    # Track statistics
                    train_loss += loss.item() * X_raw_batch.size(0)
                    _, predicted = torch.max(outputs, 1)
                    train_total += y_batch.size(0)
                    train_correct += (predicted == y_batch).sum().item()
                
                # Calculate training metrics
                train_loss = train_loss / train_total
                train_acc = train_correct / train_total
                
                # Validation phase
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                all_val_preds = []
                all_val_targets = []
                
                with torch.no_grad():
                    for batch in val_loader:
                        # Unpack batch
                        if len(batch) == 4:  # Raw, CSP, Freq, Labels
                            X_raw_batch, X_csp_batch, X_freq_batch, y_batch = [b.to(self.device) for b in batch]
                        elif len(batch) == 3:
                            if self.use_csp and not self.use_freq:  # Raw, CSP, Labels
                                X_raw_batch, X_csp_batch, y_batch = [b.to(self.device) for b in batch]
                                X_freq_batch = None
                            else:  # Raw, Freq, Labels
                                X_raw_batch, X_freq_batch, y_batch = [b.to(self.device) for b in batch]
                                X_csp_batch = None
                        else:  # Raw, Labels
                            X_raw_batch, y_batch = [b.to(self.device) for b in batch]
                            X_csp_batch = None
                            X_freq_batch = None
                        
                        # Forward pass
                        outputs = self.model(X_raw_batch, X_csp_batch, X_freq_batch)
                        loss = criterion(outputs, y_batch)
                        
                        # Track statistics
                        val_loss += loss.item() * X_raw_batch.size(0)
                        _, predicted = torch.max(outputs, 1)
                        val_total += y_batch.size(0)
                        val_correct += (predicted == y_batch).sum().item()
                        
                        # Store predictions and targets for balanced accuracy
                        all_val_preds.extend(predicted.cpu().numpy())
                        all_val_targets.extend(y_batch.cpu().numpy())
                
                # Calculate validation metrics
                val_loss = val_loss / val_total
                val_acc = val_correct / val_total
                val_balanced_acc = balanced_accuracy_score(all_val_targets, all_val_preds)
                
                # Update learning rate scheduler
                scheduler.step(val_loss)
                
                # Update history
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['train_acc'].append(train_acc)
                history['val_acc'].append(val_acc)
                history['val_balanced_acc'].append(val_balanced_acc)
                
                # Print progress
                print(f"Epoch {epoch+1}/{self.n_epochs} - "
                      f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
                      f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, "
                      f"val_balanced_acc: {val_balanced_acc:.4f}")
                
                # Check for improvement
                if val_balanced_acc > best_balanced_acc:
                    best_balanced_acc = val_balanced_acc
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                    print(f"  ✓ New best model (balanced acc: {best_balanced_acc:.4f})")
                else:
                    patience_counter += 1
                    print(f"  × No improvement for {patience_counter} epochs "
                          f"(best balanced acc: {best_balanced_acc:.4f})")
                    
                    # Early stopping
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
        
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user at epoch {}/{}".format(epoch+1, self.n_epochs))
            print("Evaluating best model found so far...")
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Loaded best model with balanced accuracy: {best_balanced_acc:.4f}")
        
        # Final evaluation on validation set
        eval_results = self.evaluate(X_raw, y, X_csp, X_freq, subset='val', val_indices=val_indices)
        
        # Add history to results
        eval_results['history'] = history
        
        return eval_results
    
    def evaluate(self, X_raw, y, X_csp=None, X_freq=None, subset='all', val_indices=None):
        """
        Evaluate the model on new data
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        # Normalize data
        X_raw = subject_specific_normalize(X_raw)
        
        # Normalize CSP and frequency features if provided
        if X_csp is not None:
            # Standardize CSP features
            X_csp = (X_csp - X_csp.mean(axis=0)) / (X_csp.std(axis=0) + 1e-8)
        
        if X_freq is not None:
            # Standardize frequency features
            X_freq = (X_freq - X_freq.mean(axis=0)) / (X_freq.std(axis=0) + 1e-8)
        
        # Select subset if specified
        if subset == 'val' and val_indices is not None:
            X_raw = X_raw[val_indices]
            y = y[val_indices]
            if X_csp is not None:
                X_csp = X_csp[val_indices]
            if X_freq is not None:
                X_freq = X_freq[val_indices]
        
        # Convert to PyTorch tensors
        X_raw_tensor = torch.FloatTensor(X_raw).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Prepare CSP tensor if available
        X_csp_tensor = torch.FloatTensor(X_csp).to(self.device) if X_csp is not None else None
        
        # Prepare frequency tensor if available
        X_freq_tensor = torch.FloatTensor(X_freq).to(self.device) if X_freq is not None else None
        
        # Create dataset and dataloader
        if X_csp_tensor is not None and X_freq_tensor is not None:
            # Dataset with all features
            dataset = TensorDataset(X_raw_tensor, X_csp_tensor, X_freq_tensor, y_tensor)
        elif X_csp_tensor is not None:
            # Dataset with raw and CSP features
            dataset = TensorDataset(X_raw_tensor, X_csp_tensor, y_tensor)
        elif X_freq_tensor is not None:
            # Dataset with raw and frequency features
            dataset = TensorDataset(X_raw_tensor, X_freq_tensor, y_tensor)
        else:
            # Dataset with only raw features
            dataset = TensorDataset(X_raw_tensor, y_tensor)
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Evaluate
        self.model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Unpack batch
                if len(batch) == 4:  # Raw, CSP, Freq, Labels
                    X_raw_batch, X_csp_batch, X_freq_batch, y_batch = batch
                elif len(batch) == 3:
                    if self.use_csp and not self.use_freq:  # Raw, CSP, Labels
                        X_raw_batch, X_csp_batch, y_batch = batch
                        X_freq_batch = None
                    else:  # Raw, Freq, Labels
                        X_raw_batch, X_freq_batch, y_batch = batch
                        X_csp_batch = None
                else:  # Raw, Labels
                    X_raw_batch, y_batch = batch
                    X_csp_batch = None
                    X_freq_batch = None
                
                # Forward pass
                outputs = self.model(X_raw_batch, X_csp_batch, X_freq_batch)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                # Store results
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        
        accuracy = accuracy_score(all_targets, all_preds)
        balanced_acc = balanced_accuracy_score(all_targets, all_preds)
        cm = confusion_matrix(all_targets, all_preds)
        
        # Calculate per-class metrics
        class_accuracies = {}
        for i in range(self.n_classes):
            mask = all_targets == i
            if np.sum(mask) > 0:
                class_accuracies[i] = accuracy_score(all_targets[mask], all_preds[mask])
            else:
                class_accuracies[i] = float('nan')
        
        print(f"\nEvaluation results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Balanced accuracy: {balanced_acc:.4f}")
        print(f"Confusion matrix:\n{cm}")
        
        for i in range(self.n_classes):
            print(f"Class {i} accuracy: {class_accuracies.get(i, float('nan')):.4f}")
        
        return {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'targets': all_targets,
            'probabilities': all_probs,
            'class_accuracies': class_accuracies
        }


def train_hybrid_model(subject_id=SUBJECT_ID, run_ids=RUN_IDS, save_model=True, variable_length=False):
    """
    Train the hybrid CNN/Transformer model on a single subject's data
    """
    # Load data
    X, y, info = load_subject_data(subject_id, run_ids)
    
    # Print dataset information
    print(f"\nTraining Hybrid CNN/Transformer for subject {subject_id}")
    print(f"Data shape: {X.shape} (trials, channels, time points)")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Create classifier
    classifier = HybridModelClassifier(
        n_classes=len(np.unique(y)),
        embedding_dim=64,
        n_heads=4,
        n_layers=2,
        dropout=0.3,
        lr=0.0005,
        batch_size=16,
        n_epochs=200,
        weight_decay=0.01,
        augmentation_factor=2,
        use_csp=True,
        use_freq=True,
        n_csp_components=4,
        variable_length_augmentation=variable_length
    )
    
    # Train and evaluate
    results = classifier.train_and_evaluate(X, y, val_split=0.2, early_stopping_patience=30)
    
    # Add the trained classifier to the results
    results['classifier'] = classifier
    
    # Plot training history
    plt.figure(figsize=(15, 10))
    
    # Loss plot
    plt.subplot(2, 2, 1)
    plt.plot(results['history']['train_loss'], 'b-', label='Training Loss')
    plt.plot(results['history']['val_loss'], 'r-', label='Validation Loss')
    plt.title(f'Training and Validation Loss - Subject {subject_id}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(2, 2, 2)
    plt.plot(results['history']['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(results['history']['val_acc'], 'r-', label='Validation Accuracy')
    plt.plot(results['history']['val_balanced_acc'], 'g-', label='Validation Balanced Accuracy')
    plt.title(f'Training and Validation Accuracy - Subject {subject_id}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Confusion Matrix
    plt.subplot(2, 2, 3)
    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - Subject {subject_id}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Final predictions vs. actual
    plt.subplot(2, 2, 4)
    plt.scatter(range(len(results['targets'])), results['targets'], c='blue', alpha=0.5, label='Actual')
    plt.scatter(range(len(results['predictions'])), results['predictions'], c='red', alpha=0.5, label='Predicted')
    plt.title('Validation Set: Actual vs. Predicted')
    plt.xlabel('Sample Index')
    plt.ylabel('Class')
    plt.yticks(np.unique(results['targets']))
    plt.legend()
    
    plt.tight_layout()
    
    # Save the model if requested
    if save_model:
        model_dir = os.path.join(os.getcwd(), 'models')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f'hybrid_model_subject_{subject_id}_runs_{"-".join(run_ids)}.pt')
        classifier.save_model(model_path)
    
    return results


def train_multi_subject_hybrid_model(subject_ids=None, run_ids=RUN_IDS, save_model=True, variable_length=False, max_subjects=None):
    """
    Train the hybrid CNN/Transformer model on data combined from multiple subjects
    
    Args:
        subject_ids: Specific subject IDs to use (None = all available)
        run_ids: Run IDs to use
        save_model: Whether to save the trained model
        variable_length: Whether to use variable-length augmentation
        max_subjects: Maximum number of subjects to include (None = all)
    """
    # Load combined data with subject sampling if specified
    X, y, subject_indices = load_multi_subject_data(subject_ids, run_ids, max_subjects=max_subjects)
    
    # Get number of unique subjects
    n_subjects = len(np.unique(subject_indices))
    
    # Print dataset information
    print(f"\nTraining Hybrid CNN/Transformer on combined data from {n_subjects} subjects")
    print(f"Data shape: {X.shape} (trials, channels, time points)")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Apply subject-specific normalization
    print("Applying subject-specific normalization...")
    X = subject_specific_normalize(X, subject_indices)
    
    # Create classifier
    classifier = HybridModelClassifier(
        n_classes=len(np.unique(y)),
        embedding_dim=64,
        n_heads=4,
        n_layers=2,
        dropout=0.3,
        lr=0.0005,
        batch_size=32,  # Larger batch size for more data
        n_epochs=200,
        weight_decay=0.01,
        augmentation_factor=2,
        use_csp=True,
        use_freq=True,
        n_csp_components=6,  # More components for multi-subject
        variable_length_augmentation=variable_length
    )
    
    # Train and evaluate
    results = classifier.train_and_evaluate(X, y, val_split=0.2, early_stopping_patience=30)
    
    # Add the trained classifier to the results
    results['classifier'] = classifier
    
    # Plot training history
    plt.figure(figsize=(15, 10))
    
    # Loss plot
    plt.subplot(2, 2, 1)
    plt.plot(results['history']['train_loss'], 'b-', label='Training Loss')
    plt.plot(results['history']['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss - Multi-Subject')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(2, 2, 2)
    plt.plot(results['history']['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(results['history']['val_acc'], 'r-', label='Validation Accuracy')
    plt.plot(results['history']['val_balanced_acc'], 'g-', label='Validation Balanced Accuracy')
    plt.title('Training and Validation Accuracy - Multi-Subject')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Confusion Matrix
    plt.subplot(2, 2, 3)
    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Multi-Subject')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Final predictions vs. actual
    plt.subplot(2, 2, 4)
    plt.scatter(range(len(results['targets'])), results['targets'], c='blue', alpha=0.5, label='Actual')
    plt.scatter(range(len(results['predictions'])), results['predictions'], c='red', alpha=0.5, label='Predicted')
    plt.title('Validation Set: Actual vs. Predicted')
    plt.xlabel('Sample Index')
    plt.ylabel('Class')
    plt.yticks(np.unique(results['targets']))
    plt.legend()
    
    plt.tight_layout()
    
    # Save the model if requested
    if save_model:
        model_dir = os.path.join(os.getcwd(), 'models')
        os.makedirs(model_dir, exist_ok=True)
        subject_str = 'all' if subject_ids is None else f"{len(subject_ids)}_subjects"
        if max_subjects:
            subject_str = f"{min(n_subjects, max_subjects)}_subjects_sampled"
        model_path = os.path.join(model_dir, f'hybrid_model_{subject_str}_runs_{"-".join(run_ids)}.pt')
        classifier.save_model(model_path)
    
    return results


def evaluate_augmentation_methods(subject_id=SUBJECT_ID, run_ids=RUN_IDS, multi_subject=False, 
                              n_epochs=50, n_repeats=3, save_results=True,
                              evaluation_mode="individual", max_subjects=None):
    """
    Systematically evaluate the impact of different data augmentation methods on
    model performance for single or multi-subject scenarios.
    
    Args:
        subject_id: Subject ID for single-subject evaluation
        run_ids: Run IDs to use
        multi_subject: Whether to evaluate on multiple subjects
        n_epochs: Number of epochs for each training run (reduced for faster evaluation)
        n_repeats: Number of repetitions for each method (to account for randomness)
        save_results: Whether to save the results to a file
        evaluation_mode: Either "individual" (test each method alone) or 
                         "leave_one_out" (test all methods except one)
        max_subjects: Maximum number of subjects to include for multi-subject evaluation
        
    Returns:
        Dictionary containing the evaluation results
    """
    # Load data based on scenario
    if multi_subject:
        print(f"Loading multi-subject data...")
        X, y, subject_indices = load_multi_subject_data(run_ids=run_ids, max_subjects=max_subjects)
        # Apply subject-specific normalization
        X = subject_specific_normalize(X, subject_indices)
        n_subjects = len(np.unique(subject_indices))
        title_prefix = f"Multi-Subject ({n_subjects} subjects)"
        file_prefix = f"multi_{n_subjects}_subjects"
    else:
        print(f"Loading single-subject data for subject {subject_id}...")
        X, y, _ = load_subject_data(subject_id=subject_id, run_ids=run_ids)
        subject_indices = None
        title_prefix = f"Subject {subject_id}"
        file_prefix = f"sub_{subject_id}"
    
    print(f"Data shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Define all augmentation methods available
    all_methods = [
        'add_noise',
        'scale_amplitude',
        'chunk_selection',
        'variable_length_sequence',
        'frequency_band_noise',
        'spectral_perturbation',
        'smooth_warping',
        'trial_mixup'
    ]
    
    # Define augmentation methods to evaluate based on the evaluation mode
    if evaluation_mode == "individual":
        # Test each method individually
        augmentation_methods = [
            {'name': 'None', 'method': None},  # Baseline (no augmentation)
            {'name': 'Noise', 'method': 'add_noise'},
            {'name': 'Scale', 'method': 'scale_amplitude'},
            {'name': 'Chunk', 'method': 'chunk_selection'},
            {'name': 'VarLength', 'method': 'variable_length_sequence'},
            {'name': 'FreqNoise', 'method': 'frequency_band_noise'},
            {'name': 'SpectralPert', 'method': 'spectral_perturbation'},
            {'name': 'SmoothWarp', 'method': 'smooth_warping'},
            {'name': 'Mixup', 'method': 'trial_mixup'},
            {'name': 'All', 'method': 'all'}  # All methods combined
        ]
        mode_suffix = "individual"
    elif evaluation_mode == "leave_one_out":
        # Test the impact of removing each method (leave-one-out)
        augmentation_methods = [
            {'name': 'All', 'method': 'all'},  # All methods (baseline for leave-one-out)
            {'name': 'No Noise', 'method': 'no_noise', 'exclude': 'add_noise'},
            {'name': 'No Scale', 'method': 'no_scale', 'exclude': 'scale_amplitude'},
            {'name': 'No Chunk', 'method': 'no_chunk', 'exclude': 'chunk_selection'},
            {'name': 'No VarLength', 'method': 'no_varlength', 'exclude': 'variable_length_sequence'},
            {'name': 'No FreqNoise', 'method': 'no_freqnoise', 'exclude': 'frequency_band_noise'},
            {'name': 'No SpectralPert', 'method': 'no_spectralpert', 'exclude': 'spectral_perturbation'},
            {'name': 'No SmoothWarp', 'method': 'no_smoothwarp', 'exclude': 'smooth_warping'},
            {'name': 'No Mixup', 'method': 'no_mixup', 'exclude': 'trial_mixup'},
            {'name': 'None', 'method': None}  # No augmentation (for reference)
        ]
        mode_suffix = "leave_one_out"
    else:
        raise ValueError(f"Unknown evaluation mode: {evaluation_mode}")
    
    # Create a custom augmenter class for method isolation or exclusion
    class CustomAugmenter(EEGDataAugmenter):
        def __init__(self, method_name=None, exclude_method=None, noise_level=0.03, 
                     scale_range=(0.9, 1.1), chunk_max_offset=30):
            super().__init__(noise_level, scale_range, chunk_max_offset, support_variable_length=False)
            self.method_name = method_name
            self.exclude_method = exclude_method
            
            # Available augmentation methods
            self.available_methods = {
                'add_noise': self.add_noise,
                'scale_amplitude': self.scale_amplitude,
                'chunk_selection': self.chunk_selection,
                'variable_length_sequence': self.variable_length_sequence,
                'frequency_band_noise': self.frequency_band_noise,
                'spectral_perturbation': self.spectral_perturbation,
                'smooth_warping': self.smooth_warping
                # trial_mixup is handled separately
            }
            
        def augment(self, x, y=None):
            """Apply specific augmentation method(s) based on the evaluation mode"""
            # Handle the None case (no augmentation)
            if self.method_name is None:
                return x.copy()
                
            # Individual method mode
            if self.exclude_method is None:
                if self.method_name == 'add_noise':
                    return self.add_noise(x.copy())
                elif self.method_name == 'scale_amplitude':
                    return self.scale_amplitude(x.copy())
                elif self.method_name == 'chunk_selection':
                    return self.chunk_selection(x.copy())
                elif self.method_name == 'variable_length_sequence':
                    return self.variable_length_sequence(x.copy())
                elif self.method_name == 'frequency_band_noise':
                    return self.frequency_band_noise(x.copy())
                elif self.method_name == 'spectral_perturbation':
                    return self.spectral_perturbation(x.copy())
                elif self.method_name == 'smooth_warping':
                    return self.smooth_warping(x.copy())
                elif self.method_name == 'trial_mixup':
                    return self.trial_mixup(x.copy(), y)
                elif self.method_name == 'all':
                    # Apply 2-3 random augmentations from the full list
                    x_aug = x.copy()
                    
                    # Get all available augmentation methods
                    augmentations = list(self.available_methods.values())
                    
                    # Apply 2-3 random augmentations
                    num_augmentations = np.random.randint(2, 4)
                    selected_augmentations = np.random.choice(augmentations, 
                                                            size=min(num_augmentations, len(augmentations)), 
                                                            replace=False)
                    
                    # Apply selected augmentations
                    for augmentation in selected_augmentations:
                        x_aug = augmentation(x_aug)
                    
                    # Separately decide whether to apply mixup (with 30% probability)
                    if self.trial_cache is not None and np.random.rand() < 0.3:
                        x_aug = self.trial_mixup(x_aug, y, same_class_only=True)
                        
                    return x_aug
                else:
                    return x.copy()
            # Leave-one-out mode
            else:
                x_aug = x.copy()
                
                # Create a copy of available methods and remove the excluded one
                available_methods = dict(self.available_methods)
                if self.exclude_method in available_methods:
                    del available_methods[self.exclude_method]
                
                # Get the list of augmentation functions that are available
                augmentations = list(available_methods.values())
                
                # Apply 2-3 random augmentations from the filtered list
                num_augmentations = np.random.randint(2, 4)
                selected_augmentations = np.random.choice(augmentations, 
                                                        size=min(num_augmentations, len(augmentations)), 
                                                        replace=False)
                
                # Apply selected augmentations
                for augmentation in selected_augmentations:
                    x_aug = augmentation(x_aug)
                
                # Apply mixup separately (unless it's the excluded method)
                if self.exclude_method != 'trial_mixup' and self.trial_cache is not None and np.random.rand() < 0.3:
                    x_aug = self.trial_mixup(x_aug, y)
                    
                return x_aug
                
        def augment_batch(self, X, y, augmentation_factor=2):
            """Generate augmented data for a batch of EEG trials"""
            n_trials = X.shape[0]
            
            # Cache a copy of the original trials for mixup augmentation
            self.trial_cache = [X[i].copy() for i in range(n_trials)]
            if y is not None:
                self.trial_labels = y.copy()
            
            # Fixed-length mode (we're not supporting variable-length for evaluation)
            augmented_X = [X]
            augmented_y = [y]
            
            # Add augmented data
            for _ in range(augmentation_factor):
                X_batch_aug = np.array([self.augment(X[i], y[i] if y is not None else None) for i in range(n_trials)])
                augmented_X.append(X_batch_aug)
                augmented_y.append(y)
            
            # Concatenate original and augmented data
            X_augmented = np.concatenate(augmented_X, axis=0)
            y_augmented = np.concatenate(augmented_y, axis=0)
            
            return X_augmented, y_augmented
    
    # Create a fixed train/val split to ensure fair comparison
    n_classes = len(np.unique(y))
    indices_by_class = [np.where(y == i)[0] for i in range(n_classes)]
    
    train_indices = []
    val_indices = []
    
    # Use a fixed random seed for reproducibility
    np.random.seed(42)
    
    for class_indices in indices_by_class:
        np.random.shuffle(class_indices)
        val_size = int(len(class_indices) * 0.2)
        
        val_indices.extend(class_indices[:val_size])
        train_indices.extend(class_indices[val_size:])
    
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    
    # Initialize results storage
    results = {
        'method_names': [m['name'] for m in augmentation_methods],
        'val_accuracy': np.zeros((len(augmentation_methods), n_repeats)),
        'val_balanced_accuracy': np.zeros((len(augmentation_methods), n_repeats)),
        'training_time': np.zeros((len(augmentation_methods), n_repeats)),
        'best_epoch': np.zeros((len(augmentation_methods), n_repeats), dtype=int)
    }
    
    # Function to create a classifier with a specific augmenter
    def create_classifier(method_info):
        # Consistent batch size across all methods
        batch_size = 16
    
        if multi_subject:
            kwargs = {
                'n_classes': n_classes,
                'embedding_dim': 64,
                'n_heads': 4,
                'n_layers': 2,
                'dropout': 0.3,
                'lr': 0.0005,
                'batch_size': batch_size,
                'n_epochs': n_epochs,
                'weight_decay': 0.01,
                'augmentation_factor': 3,  # Increase augmentation factor for more data
                'use_csp': True,
                'use_freq': True,
                'n_csp_components': 6,
                'variable_length_augmentation': False  # Disable variable length for evaluation
            }
        else:
            kwargs = {
                'n_classes': n_classes,
                'embedding_dim': 64,
                'n_heads': 4,
                'n_layers': 2,
                'dropout': 0.3,
                'lr': 0.0005,
                'batch_size': batch_size,
                'n_epochs': n_epochs,
                'weight_decay': 0.01,
                'augmentation_factor': 3,  # Increase augmentation factor for more data
                'use_csp': True,
                'use_freq': True,
                'n_csp_components': 4,
                'variable_length_augmentation': False  # Disable variable length for evaluation
            }
        
        classifier = HybridModelClassifier(**kwargs)
        
        # Replace the augmenter with our custom version
        if evaluation_mode == "individual":
            classifier.augmenter = CustomAugmenter(method_name=method_info['method'])
        else:  # leave_one_out
            exclude = method_info.get('exclude', None)
            classifier.augmenter = CustomAugmenter(method_name=method_info['method'], 
                                                  exclude_method=exclude)
        
        # Cache the training data for mixup augmentation
        # This will happen in prepare_data
        
        return classifier
    
    # Evaluate each augmentation method
    for i, aug_method in enumerate(augmentation_methods):
        method_name = aug_method['name']
        print(f"\n{'='*50}")
        print(f"Evaluating: {method_name}")
        print(f"{'='*50}")
        
        for repeat in range(n_repeats):
            print(f"\nRepetition {repeat+1}/{n_repeats}")
            
            # Create classifier with the specific augmentation method
            classifier = create_classifier(aug_method)
            
            # Track training time
            start_time = time.time()
            
            # Train and evaluate
            try:
                train_results = classifier.train_and_evaluate(
                    X, y, val_split=0.2, early_stopping_patience=20  # Increased patience
                )
                
                # Store results (these use the best model from training)
                results['val_accuracy'][i, repeat] = train_results['accuracy']
                results['val_balanced_accuracy'][i, repeat] = train_results['balanced_accuracy']
                
                # Record the best epoch
                best_epoch = np.argmax(train_results['history']['val_balanced_acc'])
                results['best_epoch'][i, repeat] = best_epoch + 1  # +1 since epochs are 1-indexed in output
            except Exception as e:
                print(f"Error during training: {str(e)}")
                results['val_accuracy'][i, repeat] = np.nan
                results['val_balanced_accuracy'][i, repeat] = np.nan
                results['best_epoch'][i, repeat] = 0
            
            end_time = time.time()
            results['training_time'][i, repeat] = end_time - start_time
            
            print(f"Validation accuracy: {results['val_accuracy'][i, repeat]:.4f}")
            print(f"Validation balanced accuracy: {results['val_balanced_accuracy'][i, repeat]:.4f}")
            print(f"Best epoch: {results['best_epoch'][i, repeat]}")
            print(f"Training time: {results['training_time'][i, repeat]:.1f} seconds")
    
    # Calculate average and standard deviation
    results['mean_accuracy'] = np.nanmean(results['val_accuracy'], axis=1)
    results['std_accuracy'] = np.nanstd(results['val_accuracy'], axis=1)
    results['mean_balanced_accuracy'] = np.nanmean(results['val_balanced_accuracy'], axis=1)
    results['std_balanced_accuracy'] = np.nanstd(results['val_balanced_accuracy'], axis=1)
    results['mean_training_time'] = np.nanmean(results['training_time'], axis=1)
    results['mean_best_epoch'] = np.nanmean(results['best_epoch'], axis=1)
    
    # Visualize results
    plt.figure(figsize=(12, 12))
    
    # Plot accuracy
    plt.subplot(3, 1, 1)
    x = np.arange(len(results['method_names']))
    width = 0.35
    
    plt.bar(x - width/2, results['mean_accuracy'], width, 
            yerr=results['std_accuracy'], label='Accuracy', 
            color='royalblue', alpha=0.7, capsize=5)
    plt.bar(x + width/2, results['mean_balanced_accuracy'], width, 
            yerr=results['std_balanced_accuracy'], label='Balanced Accuracy', 
            color='darkorange', alpha=0.7, capsize=5)
    
    plt.xlabel('Augmentation Method')
    plt.ylabel('Validation Accuracy')
    if evaluation_mode == "leave_one_out":
        plt.title(f'{title_prefix} - Impact of Removing Each Augmentation Method')
    else:
        plt.title(f'{title_prefix} - Impact of Individual Augmentation Methods')
    plt.xticks(x, results['method_names'], rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot training time
    plt.subplot(3, 1, 2)
    plt.bar(x, results['mean_training_time'], color='forestgreen', alpha=0.7)
    plt.xlabel('Augmentation Method')
    plt.ylabel('Training Time (s)')
    plt.title(f'{title_prefix} - Training Time by Method')
    plt.xticks(x, results['method_names'], rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot best epoch
    plt.subplot(3, 1, 3)
    plt.bar(x, results['mean_best_epoch'], color='purple', alpha=0.7)
    plt.xlabel('Augmentation Method')
    plt.ylabel('Best Epoch')
    plt.title(f'{title_prefix} - Average Best Epoch by Method')
    plt.xticks(x, results['method_names'], rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save results
    if save_results:
        # Create results directory if it doesn't exist
        results_dir = os.path.join(os.getcwd(), 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save plot
        plt.savefig(os.path.join(results_dir, f'augmentation_impact_{file_prefix}_{mode_suffix}.png'))
        
        # Save numerical results
        np.savez(os.path.join(results_dir, f'augmentation_impact_{file_prefix}_{mode_suffix}.npz'), **results)
    
    
    # Print summary
    print("\nSummary of Augmentation Methods Impact:")
    print(f"{'Method':<15} {'Accuracy':<15} {'Balanced Acc':<15} {'Time (s)':<10} {'Best Epoch':<10}")
    print('-' * 65)
    
    for i, method in enumerate(results['method_names']):
        print(f"{method:<15} {results['mean_accuracy'][i]:.4f} ± {results['std_accuracy'][i]:.4f}   "
              f"{results['mean_balanced_accuracy'][i]:.4f} ± {results['std_balanced_accuracy'][i]:.4f}   "
              f"{results['mean_training_time'][i]:.1f}      {results['mean_best_epoch'][i]:.1f}")
    
    # Find the best method
    if evaluation_mode == "individual":
        best_idx = np.argmax(results['mean_balanced_accuracy'])
        best_method = results['method_names'][best_idx]
        print(f"\nBest augmentation method: {best_method} "
              f"(Balanced Accuracy: {results['mean_balanced_accuracy'][best_idx]:.4f})")
    else:  # leave_one_out
        # For leave-one-out, the worst method to exclude (causing biggest drop in performance) is most important
        all_idx = results['method_names'].index('All')
        all_acc = results['mean_balanced_accuracy'][all_idx]
        
        # Calculate performance drops when excluding each method
        drops = []
        for i, name in enumerate(results['method_names']):
            if name.startswith('No '):
                drop = all_acc - results['mean_balanced_accuracy'][i]
                drops.append((name[3:], drop))  # Remove "No " prefix
        
        # Sort by largest performance drop
        drops.sort(key=lambda x: x[1], reverse=True)
        
        # Print the most important methods (those with largest drop when excluded)
        print("\nMost important augmentation methods (ranked by performance drop when excluded):")
        for method, drop in drops:
            print(f"{method}: {drop:.4f} drop in balanced accuracy when excluded")
    
    return results


def visualize_augmentation_effects(subject_id=SUBJECT_ID, run_ids=RUN_IDS, n_examples=5):
    """
    Visualize the effect of each augmentation method on EEG data
    
    Args:
        subject_id: Subject ID to use
        run_ids: Run IDs to use
        n_examples: Number of example trials to visualize
    """
    # Load data
    X, y, _ = load_subject_data(subject_id=subject_id, run_ids=run_ids)
    
    # Apply normalization to better see the effects
    X = subject_specific_normalize(X)
    
    # Create augmenter
    augmenter = EEGDataAugmenter()
    
    # Get augmentation methods
    augmentation_methods = [
        {'name': 'Original', 'method': None},
        {'name': 'Noise', 'method': augmenter.add_noise},
        {'name': 'Scale', 'method': augmenter.scale_amplitude},
        {'name': 'Chunk', 'method': augmenter.chunk_selection},
        {'name': 'VarLength', 'method': augmenter.variable_length_sequence},
        {'name': 'FreqNoise', 'method': augmenter.frequency_band_noise},
        {'name': 'SpectralPert', 'method': augmenter.spectral_perturbation},
        {'name': 'SmoothWarp', 'method': augmenter.smooth_warping}
        # Mixup requires multiple trials, handled separately below
    ]
    
    # Cache some trials for mixup visualization
    augmenter.trial_cache = [X[i].copy() for i in range(min(20, len(X)))]
    augmenter.trial_labels = y[:min(20, len(y))].copy()
    
    # Select random examples
    n_trials = X.shape[0]
    example_indices = np.random.choice(n_trials, size=n_examples, replace=False)
    
    for idx in example_indices:
        # Get original sample
        x_orig = X[idx]
        label = y[idx]
        
        # Apply each augmentation method
        augmented_samples = [x_orig.copy()]
        for method in augmentation_methods[1:]:
            if method['method'] is not None:
                augmented_samples.append(method['method'](x_orig.copy()))
        
        # Add mixup as a special case
        if augmenter.trial_cache is not None:
            augmented_samples.append(augmenter.trial_mixup(x_orig.copy(), label))
            method_names = [m['name'] for m in augmentation_methods] + ['Mixup']
        else:
            method_names = [m['name'] for m in augmentation_methods]
        
        # Plot
        plt.figure(figsize=(15, 12))
        
        # Plot central electrode (assuming C3 or similar is in the middle)
        center_ch = x_orig.shape[0] // 2
        
        for i, (x_aug, method_name) in enumerate(zip(augmented_samples, method_names)):
            plt.subplot(len(augmented_samples), 1, i+1)
            plt.plot(x_aug[center_ch])
            plt.title(f"{method_name}")
            plt.ylabel('Amplitude')
            
            # For first subplot add more details
            if i == 0:
                plt.title(f"Original Signal (Class {label}, Trial {idx}, Channel {center_ch})")
        
        plt.xlabel('Time points')
        plt.tight_layout()
        
        # Plot frequency domain representation
        plt.figure(figsize=(15, 12))
        
        for i, (x_aug, method_name) in enumerate(zip(augmented_samples, method_names)):
            plt.subplot(len(augmented_samples), 1, i+1)
            
            # Apply FFT
            x_fft = np.abs(np.fft.rfft(x_aug[center_ch]))
            freqs = np.fft.rfftfreq(x_aug.shape[1], d=1/160)  # Assuming 160Hz sampling rate
            
            plt.semilogy(freqs, x_fft)
            plt.title(f"{method_name} - Frequency Domain")
            plt.ylabel('Log Power')
            
            # Highlight mu and beta bands
            plt.axvspan(8, 12, color='yellow', alpha=0.3, label='μ band')
            plt.axvspan(13, 30, color='green', alpha=0.3, label='β band')
            
            if i == 0:
                plt.legend()
        
        plt.xlabel('Frequency (Hz)')
        plt.tight_layout()


def evaluate_optimized_augmentation(subject_id=SUBJECT_ID, run_ids=RUN_IDS, multi_subject=True, 
                               n_epochs=150, n_repeats=5, save_results=True, max_subjects=None):
    """
    Evaluate a custom optimized augmentation strategy that uses only the beneficial methods
    based on leave-one-out analysis results.
    
    Args:
        subject_id: Subject ID for single-subject evaluation
        run_ids: Run IDs to use
        multi_subject: Whether to evaluate on multiple subjects
        n_epochs: Number of epochs for each training run
        n_repeats: Number of repetitions for each method
        save_results: Whether to save the results to a file
        max_subjects: Maximum number of subjects to include (None = all)
        
    Returns:
        Dictionary containing the evaluation results
    """
    # Load data based on scenario
    if multi_subject:
        print(f"Loading multi-subject data...")
        X, y, subject_indices = load_multi_subject_data(run_ids=run_ids, max_subjects=max_subjects)
        # Apply subject-specific normalization
        X = subject_specific_normalize(X, subject_indices)
        n_subjects = len(np.unique(subject_indices))
        title_prefix = f"Multi-Subject ({n_subjects} subjects)"
        file_prefix = f"multi_{n_subjects}_subjects"
    else:
        print(f"Loading single-subject data for subject {subject_id}...")
        X, y, _ = load_subject_data(subject_id=subject_id, run_ids=run_ids)
        subject_indices = None
        title_prefix = f"Subject {subject_id}"
        file_prefix = f"sub_{subject_id}"
    
    print(f"Data shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Define augmentation methods to evaluate
    augmentation_methods = [
        {'name': 'None', 'method': None},  # Baseline (no augmentation)
        {'name': 'Optimized', 'method': 'optimized'},  # Our optimized strategy
        {'name': 'All', 'method': 'all'}  # All methods (for comparison)
    ]
    
    # Create a custom augmenter class for optimized augmentation
    class OptimizedAugmenter(EEGDataAugmenter):
        def __init__(self, method_name=None, noise_level=0.03, 
                     scale_range=(0.9, 1.1), chunk_max_offset=30):
            super().__init__(noise_level, scale_range, chunk_max_offset, support_variable_length=False)
            self.method_name = method_name
            
            # Only beneficial augmentation methods
            self.beneficial_methods = {
                'add_noise': self.add_noise,
                'scale_amplitude': self.scale_amplitude,
                'spectral_perturbation': self.spectral_perturbation,
                'smooth_warping': self.smooth_warping
                # trial_mixup is handled separately
            }
            
            # All available methods (for comparison)
            self.all_methods = {
                'add_noise': self.add_noise,
                'scale_amplitude': self.scale_amplitude,
                'chunk_selection': self.chunk_selection,
                'variable_length_sequence': self.variable_length_sequence,
                'frequency_band_noise': self.frequency_band_noise,
                'spectral_perturbation': self.spectral_perturbation,
                'smooth_warping': self.smooth_warping
                # trial_mixup is handled separately
            }
            
        def augment(self, x, y=None):
            """Apply specific augmentation method(s) based on the strategy"""
            # Handle the None case (no augmentation)
            if self.method_name is None:
                return x.copy()
                
            # Optimized strategy
            if self.method_name == 'optimized':
                x_aug = x.copy()
                
                # Get beneficial augmentation methods
                augmentations = list(self.beneficial_methods.values())
                
                # Apply 1-2 random augmentations (slightly fewer since we've curated the list)
                num_augmentations = np.random.randint(1, 3)
                selected_augmentations = np.random.choice(augmentations, 
                                                        size=min(num_augmentations, len(augmentations)), 
                                                        replace=False)
                
                # Apply selected augmentations
                for augmentation in selected_augmentations:
                    x_aug = augmentation(x_aug)
                
                # Apply mixup with higher probability (50%) since it's the most beneficial
                if self.trial_cache is not None and np.random.rand() < 0.5:
                    x_aug = self.trial_mixup(x_aug, y, same_class_only=True)
                    
                return x_aug
                
            # All methods (for comparison)
            elif self.method_name == 'all':
                x_aug = x.copy()
                
                # Get all available augmentation methods
                augmentations = list(self.all_methods.values())
                
                # Apply 2-3 random augmentations
                num_augmentations = np.random.randint(2, 4)
                selected_augmentations = np.random.choice(augmentations, 
                                                        size=min(num_augmentations, len(augmentations)), 
                                                        replace=False)
                
                # Apply selected augmentations
                for augmentation in selected_augmentations:
                    x_aug = augmentation(x_aug)
                
                # Apply mixup with standard probability (30%)
                if self.trial_cache is not None and np.random.rand() < 0.3:
                    x_aug = self.trial_mixup(x_aug, y, same_class_only=True)
                    
                return x_aug
            else:
                return x.copy()
                
        def augment_batch(self, X, y, augmentation_factor=3):
            """Generate augmented data for a batch of EEG trials"""
            n_trials = X.shape[0]
            
            # Cache a copy of the original trials for mixup augmentation
            self.trial_cache = [X[i].copy() for i in range(n_trials)]
            if y is not None:
                self.trial_labels = y.copy()
            
            # Fixed-length mode
            augmented_X = [X]
            augmented_y = [y]
            
            # Add augmented data
            for _ in range(augmentation_factor):
                X_batch_aug = np.array([self.augment(X[i], y[i] if y is not None else None) for i in range(n_trials)])
                augmented_X.append(X_batch_aug)
                augmented_y.append(y)
            
            # Concatenate original and augmented data
            X_augmented = np.concatenate(augmented_X, axis=0)
            y_augmented = np.concatenate(augmented_y, axis=0)
            
            return X_augmented, y_augmented
    
    # Create a fixed train/val split to ensure fair comparison
    n_classes = len(np.unique(y))
    indices_by_class = [np.where(y == i)[0] for i in range(n_classes)]
    
    train_indices = []
    val_indices = []
    
    # Use a fixed random seed for reproducibility
    np.random.seed(42)
    
    for class_indices in indices_by_class:
        np.random.shuffle(class_indices)
        val_size = int(len(class_indices) * 0.2)
        
        val_indices.extend(class_indices[:val_size])
        train_indices.extend(class_indices[val_size:])
    
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    
    # Initialize results storage
    results = {
        'method_names': [m['name'] for m in augmentation_methods],
        'val_accuracy': np.zeros((len(augmentation_methods), n_repeats)),
        'val_balanced_accuracy': np.zeros((len(augmentation_methods), n_repeats)),
        'training_time': np.zeros((len(augmentation_methods), n_repeats)),
        'best_epoch': np.zeros((len(augmentation_methods), n_repeats), dtype=int)
    }
    
    # Function to create a classifier with the optimized augmenter
    def create_classifier(method_info):
        # Consistent batch size across all methods
        batch_size = 24  # Increased batch size for more stability
    
        if multi_subject:
            kwargs = {
                'n_classes': n_classes,
                'embedding_dim': 64,
                'n_heads': 4,
                'n_layers': 2,
                'dropout': 0.3,
                'lr': 0.0004,  # Slightly lower learning rate
                'batch_size': batch_size,
                'n_epochs': n_epochs,
                'weight_decay': 0.01,
                'augmentation_factor': 3,  # Enhanced augmentation
                'use_csp': True,
                'use_freq': True,
                'n_csp_components': 6,
                'variable_length_augmentation': False
            }
        else:
            kwargs = {
                'n_classes': n_classes,
                'embedding_dim': 64,
                'n_heads': 4,
                'n_layers': 2,
                'dropout': 0.3,
                'lr': 0.0004,  # Slightly lower learning rate
                'batch_size': batch_size,
                'n_epochs': n_epochs,
                'weight_decay': 0.01,
                'augmentation_factor': 3,  # Enhanced augmentation
                'use_csp': True,
                'use_freq': True,
                'n_csp_components': 4,
                'variable_length_augmentation': False
            }
        
        classifier = HybridModelClassifier(**kwargs)
        
        # Replace the augmenter with our optimized version
        classifier.augmenter = OptimizedAugmenter(method_name=method_info['method'])
        
        return classifier
    
    # Evaluate each augmentation strategy
    for i, aug_method in enumerate(augmentation_methods):
        method_name = aug_method['name']
        print(f"\n{'='*50}")
        print(f"Evaluating: {method_name}")
        print(f"{'='*50}")
        
        for repeat in range(n_repeats):
            print(f"\nRepetition {repeat+1}/{n_repeats}")
            
            # Create classifier with the specific augmentation method
            classifier = create_classifier(aug_method)
            
            # Track training time
            start_time = time.time()
            
            # Train and evaluate
            try:
                train_results = classifier.train_and_evaluate(
                    X, y, val_split=0.2, early_stopping_patience=35  # Increased patience
                )
                
                # Store results (these use the best model from training)
                results['val_accuracy'][i, repeat] = train_results['accuracy']
                results['val_balanced_accuracy'][i, repeat] = train_results['balanced_accuracy']
                
                # Record the best epoch
                best_epoch = np.argmax(train_results['history']['val_balanced_acc'])
                results['best_epoch'][i, repeat] = best_epoch + 1  # +1 since epochs are 1-indexed in output
            except Exception as e:
                print(f"Error during training: {str(e)}")
                results['val_accuracy'][i, repeat] = np.nan
                results['val_balanced_accuracy'][i, repeat] = np.nan
                results['best_epoch'][i, repeat] = 0
            
            end_time = time.time()
            results['training_time'][i, repeat] = end_time - start_time
            
            print(f"Validation accuracy: {results['val_accuracy'][i, repeat]:.4f}")
            print(f"Validation balanced accuracy: {results['val_balanced_accuracy'][i, repeat]:.4f}")
            print(f"Best epoch: {results['best_epoch'][i, repeat]}")
            print(f"Training time: {results['training_time'][i, repeat]:.1f} seconds")
    
    # Calculate average and standard deviation
    results['mean_accuracy'] = np.nanmean(results['val_accuracy'], axis=1)
    results['std_accuracy'] = np.nanstd(results['val_accuracy'], axis=1)
    results['mean_balanced_accuracy'] = np.nanmean(results['val_balanced_accuracy'], axis=1)
    results['std_balanced_accuracy'] = np.nanstd(results['val_balanced_accuracy'], axis=1)
    results['mean_training_time'] = np.nanmean(results['training_time'], axis=1)
    results['mean_best_epoch'] = np.nanmean(results['best_epoch'], axis=1)
    
    # Visualize results
    plt.figure(figsize=(12, 12))
    
    # Plot accuracy
    plt.subplot(3, 1, 1)
    x = np.arange(len(results['method_names']))
    width = 0.35
    
    plt.bar(x - width/2, results['mean_accuracy'], width, 
            yerr=results['std_accuracy'], label='Accuracy', 
            color='royalblue', alpha=0.7, capsize=5)
    plt.bar(x + width/2, results['mean_balanced_accuracy'], width, 
            yerr=results['std_balanced_accuracy'], label='Balanced Accuracy', 
            color='darkorange', alpha=0.7, capsize=5)
    
    plt.xlabel('Augmentation Strategy')
    plt.ylabel('Validation Accuracy')
    plt.title(f'{title_prefix} - Optimized vs. Standard Augmentation')
    plt.xticks(x, results['method_names'], rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot training time
    plt.subplot(3, 1, 2)
    plt.bar(x, results['mean_training_time'], color='forestgreen', alpha=0.7)
    plt.xlabel('Augmentation Strategy')
    plt.ylabel('Training Time (s)')
    plt.title(f'{title_prefix} - Training Time by Strategy')
    plt.xticks(x, results['method_names'], rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot best epoch
    plt.subplot(3, 1, 3)
    plt.bar(x, results['mean_best_epoch'], color='purple', alpha=0.7)
    plt.xlabel('Augmentation Strategy')
    plt.ylabel('Best Epoch')
    plt.title(f'{title_prefix} - Average Best Epoch by Strategy')
    plt.xticks(x, results['method_names'], rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save results
    if save_results:
        # Create results directory if it doesn't exist
        results_dir = os.path.join(os.getcwd(), 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save plot
        plt.savefig(os.path.join(results_dir, f'optimized_augmentation_{file_prefix}.png'))
        
        # Save numerical results
        np.savez(os.path.join(results_dir, f'optimized_augmentation_{file_prefix}.npz'), **results)
    
    # Print summary
    print("\nSummary of Augmentation Strategies:")
    print(f"{'Strategy':<15} {'Accuracy':<15} {'Balanced Acc':<15} {'Time (s)':<10} {'Best Epoch':<10}")
    print('-' * 65)
    
    for i, method in enumerate(results['method_names']):
        print(f"{method:<15} {results['mean_accuracy'][i]:.4f} ± {results['std_accuracy'][i]:.4f}   "
              f"{results['mean_balanced_accuracy'][i]:.4f} ± {results['std_balanced_accuracy'][i]:.4f}   "
              f"{results['mean_training_time'][i]:.1f}      {results['mean_best_epoch'][i]:.1f}")
    
    # Identify best strategy
    best_idx = np.argmax(results['mean_balanced_accuracy'])
    best_method = results['method_names'][best_idx]
    print(f"\nBest augmentation strategy: {best_method} "
          f"(Balanced Accuracy: {results['mean_balanced_accuracy'][best_idx]:.4f})")
    
    return results


def train_base_model_multi_subject(run_ids=RUN_IDS, n_subjects=50, save_model=True,
                                n_epochs=200, model_name="base_model", subject_ids=None):
    """
    Train a base model on a large group of subjects using the optimized augmentation strategy.
    This serves as the pretrained model for subsequent fine-tuning.
    
    Args:
        run_ids: Run IDs to use
        n_subjects: Number of subjects to include in training (50-80 recommended)
        save_model: Whether to save the trained model
        n_epochs: Number of epochs for training
        model_name: Name to use when saving the model
        subject_ids: Specific subject IDs to use (if None, randomly samples subjects)
        
    Returns:
        Trained classifier and evaluation results
    """
    print(f"Training base model on up to {n_subjects} subjects...")
    
    # Load multi-subject data with specific subject IDs if provided
    X, y, subject_indices = load_multi_subject_data(
        subject_ids=subject_ids, 
        run_ids=run_ids, 
        max_subjects=n_subjects
    )
    
    # Get the actual subject IDs used in training
    unique_subject_indices = np.unique(subject_indices)
    if subject_ids is not None:
        # If specific subjects were provided, get the ones actually used
        used_subject_ids = [subject_ids[i] for i in unique_subject_indices]
    else:
        # If subjects were randomly sampled, convert indices to IDs
        # Assuming indices are 0-based and subject IDs are 1-based with zero-padding
        used_subject_ids = [f"{int(i)+1:03d}" for i in unique_subject_indices]
    
    # Apply subject-specific normalization
    print("Applying subject-specific normalization...")
    X = subject_specific_normalize(X, subject_indices)
    
    n_subjects_actual = len(unique_subject_indices)
    print(f"\nTraining base model on {n_subjects_actual} subjects")
    print(f"Subjects included: {used_subject_ids}")
    print(f"Data shape: {X.shape} (trials, channels, time points)")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Create optimized augmenter
    class OptimizedAugmenter(EEGDataAugmenter):
        def __init__(self, noise_level=0.03, scale_range=(0.9, 1.1), chunk_max_offset=30):
            super().__init__(noise_level, scale_range, chunk_max_offset, support_variable_length=False)
            
            # Only beneficial augmentation methods
            self.beneficial_methods = {
                'add_noise': self.add_noise,
                'scale_amplitude': self.scale_amplitude,
                'spectral_perturbation': self.spectral_perturbation,
                'smooth_warping': self.smooth_warping
                # trial_mixup is handled separately
            }
            
        def augment(self, x, y=None):
            """Apply optimized augmentation strategy"""
            x_aug = x.copy()
            
            # Get beneficial augmentation methods
            augmentations = list(self.beneficial_methods.values())
            
            # Apply 1-2 random augmentations (slightly fewer since we've curated the list)
            num_augmentations = np.random.randint(1, 3)
            selected_augmentations = np.random.choice(augmentations, 
                                                    size=min(num_augmentations, len(augmentations)), 
                                                    replace=False)
            
            # Apply selected augmentations
            for augmentation in selected_augmentations:
                x_aug = augmentation(x_aug)
            
            # Apply mixup with higher probability (50%) since it's the most beneficial
            if self.trial_cache is not None and np.random.rand() < 0.5:
                x_aug = self.trial_mixup(x_aug, y, same_class_only=True)
                
            return x_aug
                
        def augment_batch(self, X, y, augmentation_factor=3):
            """Generate augmented data for a batch of EEG trials"""
            n_trials = X.shape[0]
            
            # Cache a copy of the original trials for mixup augmentation
            self.trial_cache = [X[i].copy() for i in range(n_trials)]
            if y is not None:
                self.trial_labels = y.copy()
            
            # Fixed-length mode
            augmented_X = [X]
            augmented_y = [y]
            
            # Add augmented data
            for _ in range(augmentation_factor):
                X_batch_aug = np.array([self.augment(X[i], y[i] if y is not None else None) for i in range(n_trials)])
                augmented_X.append(X_batch_aug)
                augmented_y.append(y)
            
            # Concatenate original and augmented data
            X_augmented = np.concatenate(augmented_X, axis=0)
            y_augmented = np.concatenate(augmented_y, axis=0)
            
            return X_augmented, y_augmented
    
    # Create classifier with optimized hyperparameters for multi-subject learning
    n_classes = len(np.unique(y))
    classifier = HybridModelClassifier(
        n_classes=n_classes,
        embedding_dim=64,
        n_heads=4,
        n_layers=2,
        dropout=0.3,
        lr=0.0004,  # Slightly lower learning rate for stability
        batch_size=24,  # Larger batch size for more data
        n_epochs=n_epochs,
        weight_decay=0.01,
        augmentation_factor=3,
        use_csp=True,
        use_freq=True,
        n_csp_components=6,  # More components for multi-subject
        variable_length_augmentation=False
    )
    
    # Replace the augmenter with our optimized version
    classifier.augmenter = OptimizedAugmenter()
    
    # Train and evaluate
    print("\nTraining base model...")
    results = classifier.train_and_evaluate(X, y, val_split=0.2, early_stopping_patience=35)
    
    # Add the trained classifier to the results
    results['classifier'] = classifier
    
    # Plot training history
    plt.figure(figsize=(15, 10))
    
    # Loss plot
    plt.subplot(2, 2, 1)
    plt.plot(results['history']['train_loss'], 'b-', label='Training Loss')
    plt.plot(results['history']['val_loss'], 'r-', label='Validation Loss')
    plt.title(f'Training and Validation Loss - Base Model ({n_subjects_actual} subjects)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(2, 2, 2)
    plt.plot(results['history']['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(results['history']['val_acc'], 'r-', label='Validation Accuracy')
    plt.plot(results['history']['val_balanced_acc'], 'g-', label='Validation Balanced Accuracy')
    plt.title(f'Training and Validation Accuracy - Base Model ({n_subjects_actual} subjects)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Confusion Matrix
    plt.subplot(2, 2, 3)
    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - Base Model ({n_subjects_actual} subjects)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Final predictions vs. actual
    plt.subplot(2, 2, 4)
    plt.scatter(range(len(results['targets'])), results['targets'], c='blue', alpha=0.5, label='Actual')
    plt.scatter(range(len(results['predictions'])), results['predictions'], c='red', alpha=0.5, label='Predicted')
    plt.title('Validation Set: Actual vs. Predicted')
    plt.xlabel('Sample Index')
    plt.ylabel('Class')
    plt.yticks(np.unique(results['targets']))
    plt.legend()
    
    plt.tight_layout()
    
    # Add subject information to metadata
    metadata = {
        'training_subjects': used_subject_ids,
        'n_subjects': n_subjects_actual,
        'requested_n_subjects': n_subjects,  # Store the requested number
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'run_ids': run_ids
    }
    
    # Save the model with metadata
    if save_model:
        model_dir = os.path.join(os.getcwd(), 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        # Use the requested number of subjects in the filename
        model_path = os.path.join(model_dir, f'{model_name}_{n_subjects}_subjects.pt')
        classifier.save_model(model_path, metadata=metadata)
        
        # Also save subjects list to a separate text file for easy reference
        subjects_path = os.path.join(model_dir, f'{model_name}_{n_subjects}_subjects_list.txt')
        with open(subjects_path, 'w') as f:
            f.write(f"# Training subjects for model: {model_name}_{n_subjects}_subjects.pt\n")
            f.write(f"# Trained on: {metadata['timestamp']}\n")
            f.write(f"# Requested subjects: {n_subjects}\n")
            f.write(f"# Actual subjects used: {n_subjects_actual}\n\n")
            for subject_id in used_subject_ids:
                f.write(f"{subject_id}\n")
        
        print(f"Base model saved to {model_path}")
        print(f"Subject list saved to {subjects_path}")
        
        # Print a warning if the actual number differs from the requested number
        if n_subjects_actual != n_subjects:
            print(f"\nWARNING: {n_subjects - n_subjects_actual} subjects were excluded during data loading.")
            print(f"Requested: {n_subjects} subjects, Actual: {n_subjects_actual} subjects")
            print("This is likely because some subjects had non-standard time dimensions.")
            print(f"The model is still saved using the requested number: {model_name}_{n_subjects}_subjects.pt")
    
    return classifier, results, metadata


def finetune_subject_specific(base_model_path, subject_id, run_ids=RUN_IDS, 
                             n_epochs=100, save_models=True):
    """
    Load a pretrained base model and fine-tune it for a specific subject.
    Tests different augmentation strategies for fine-tuning.
    
    Args:
        base_model_path: Path to the pretrained model
        subject_id: Subject ID to fine-tune on
        run_ids: Run IDs to use
        n_epochs: Number of epochs for fine-tuning
        save_models: Whether to save the fine-tuned models
        
    Returns:
        Dictionary of results for different augmentation strategies
    """
    print(f"Fine-tuning for subject {subject_id}...")
    
    # Load the base model and its metadata
    base_classifier, metadata = HybridModelClassifier.load_model(base_model_path)
    
    # Check if the subject was in the training set
    if 'training_subjects' in metadata and subject_id in metadata['training_subjects']:
        print(f"WARNING: Subject {subject_id} was used in training the base model!")
        print("This could lead to data leakage and overly optimistic results.")
        print("Consider using a different subject for fair evaluation.")
    
    # Load subject data
    X, y, info = load_subject_data(subject_id=subject_id, run_ids=run_ids)
    
    # Print dataset information
    print(f"Data shape: {X.shape} (trials, channels, time points)")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Define augmentation strategies to test for fine-tuning
    aug_strategies = [
        {
            'name': 'None', 
            'augmenter': EEGDataAugmenter(support_variable_length=False),
            'apply_augmentation': False
        },
        {
            'name': 'Optimized',
            'augmenter': OptimizedAugmenter(support_variable_length=False),
            'apply_augmentation': True
        },
        {
            'name': 'Basic',
            'augmenter': BasicAugmenter(support_variable_length=False),
            'apply_augmentation': True
        }
    ]
    
    results = {
        'strategies': [s['name'] for s in aug_strategies],
        'accuracy': [],
        'balanced_accuracy': [],
        'histories': [],
        'confusion_matrices': []
    }
    
    for strategy in aug_strategies:
        print(f"\n{'='*50}")
        print(f"Fine-tuning with {strategy['name']} augmentation")
        print(f"{'='*50}")
        
        # Load the base model
        base_classifier, _ = HybridModelClassifier.load_model(base_model_path)
        
        # Set the augmenter
        base_classifier.augmenter = strategy['augmenter']
        
        # Update classifier parameters for fine-tuning
        base_classifier.n_epochs = n_epochs
        base_classifier.lr = 0.0001  # Lower learning rate for fine-tuning
        base_classifier.batch_size = 16  # Smaller batch size for subject-specific tuning
        
        # Fine-tune on the subject's data
        print(f"Fine-tuning model...")
        train_results = base_classifier.train_and_evaluate(
            X, y, val_split=0.2, early_stopping_patience=20
        )
        
        # Store results
        results['accuracy'].append(train_results['accuracy'])
        results['balanced_accuracy'].append(train_results['balanced_accuracy'])
        results['histories'].append(train_results['history'])
        results['confusion_matrices'].append(train_results['confusion_matrix'])
        
        # Plot results for this strategy
        plt.figure(figsize=(15, 5))
        
        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(train_results['history']['train_acc'], 'b-', label='Training Accuracy')
        plt.plot(train_results['history']['val_acc'], 'r-', label='Validation Accuracy')
        plt.plot(train_results['history']['val_balanced_acc'], 'g-', label='Validation Balanced Accuracy')
        plt.title(f'Subject {subject_id} - {strategy["name"]} Augmentation')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Confusion Matrix
        plt.subplot(1, 2, 2)
        sns.heatmap(train_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {strategy["name"]} Augmentation')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.tight_layout()
        
        # Save the fine-tuned model
        if save_models:
            model_dir = os.path.join(os.getcwd(), 'models')
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f'finetuned_subject_{subject_id}_{strategy["name"]}.pt')
            base_classifier.save_model(model_path)
            print(f"Fine-tuned model saved to {model_path}")
    
    # Compare strategies
    plt.figure(figsize=(10, 6))
    x = np.arange(len(results['strategies']))
    width = 0.35
    
    plt.bar(x - width/2, results['accuracy'], width, 
            label='Accuracy', color='royalblue', alpha=0.7)
    plt.bar(x + width/2, results['balanced_accuracy'], width, 
            label='Balanced Accuracy', color='darkorange', alpha=0.7)
    
    plt.xlabel('Augmentation Strategy')
    plt.ylabel('Validation Accuracy')
    plt.title(f'Subject {subject_id} - Fine-tuning Results')
    plt.xticks(x, results['strategies'])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Print summary
    print("\nFine-tuning Results Summary:")
    print(f"{'Strategy':<15} {'Accuracy':<15} {'Balanced Accuracy':<15}")
    print('-' * 45)
    
    for i, strategy in enumerate(results['strategies']):
        print(f"{strategy:<15} {results['accuracy'][i]:.4f}          {results['balanced_accuracy'][i]:.4f}")
    
    # Find the best strategy
    best_idx = np.argmax(results['balanced_accuracy'])
    best_strategy = results['strategies'][best_idx]
    print(f"\nBest augmentation strategy for Subject {subject_id}: {best_strategy} "
          f"(Balanced Accuracy: {results['balanced_accuracy'][best_idx]:.4f})")
    
    return results


def split_subjects(total_subjects=109, test_subjects_count=10, random_seed=42):
    """
    Split available subjects into training and test sets.
    
    Args:
        total_subjects: Total number of subjects available (default 109)
        test_subjects_count: Number of subjects to reserve for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        train_subjects: List of subject IDs for training
        test_subjects: List of subject IDs for testing
    """
    np.random.seed(random_seed)
    
    # Create list of all subject IDs (001-109)
    all_subjects = [f"{i:03d}" for i in range(1, total_subjects + 1)]
    
    # Randomly select test subjects
    test_indices = np.random.choice(len(all_subjects), test_subjects_count, replace=False)
    test_subjects = [all_subjects[i] for i in test_indices]
    
    # Use remaining subjects for training
    train_subjects = [s for s in all_subjects if s not in test_subjects]
    
    print(f"Split {len(all_subjects)} subjects into {len(train_subjects)} for training and {len(test_subjects)} for testing")
    return train_subjects, test_subjects


def check_subject_in_training(model_path, subject_id):
    """
    Check if a subject was used in training a model.
    
    Args:
        model_path: Path to the saved model
        subject_id: Subject ID to check
        
    Returns:
        Boolean indicating if the subject was in the training set
    """
    _, metadata = HybridModelClassifier.load_model(model_path)
    
    if 'training_subjects' in metadata:
        return subject_id in metadata['training_subjects']
    else:
        print("Model does not have training subject information.")
        return False


class OptimizedAugmenter(EEGDataAugmenter):
    """
    Optimized augmentation strategy using only the most beneficial methods
    for EEG motor imagery data
    """
    def __init__(self, noise_level=0.03, scale_range=(0.9, 1.1), chunk_max_offset=30):
        super().__init__(noise_level, scale_range, chunk_max_offset, support_variable_length=False)
        
        # Only beneficial augmentation methods
        self.beneficial_methods = {
            'add_noise': self.add_noise,
            'scale_amplitude': self.scale_amplitude,
            'spectral_perturbation': self.spectral_perturbation,
            'smooth_warping': self.smooth_warping
        }
    
    def augment(self, x, y=None):
        """Apply optimized augmentation strategy"""
        x_aug = x.copy()
        
        # Get beneficial augmentation methods
        augmentations = list(self.beneficial_methods.values())
        
        # Apply 1-2 random augmentations
        num_augmentations = np.random.randint(1, 3)
        selected_augmentations = np.random.choice(augmentations, 
                                                size=min(num_augmentations, len(augmentations)), 
                                                replace=False)
        
        # Apply selected augmentations
        for augmentation in selected_augmentations:
            x_aug = augmentation(x_aug)
        
        # Apply mixup with higher probability (50%)
        if self.trial_cache is not None and np.random.rand() < 0.5:
            x_aug = self.trial_mixup(x_aug, y, same_class_only=True)
            
        return x_aug
    
    def augment_batch(self, X, y, augmentation_factor=2):
        """
        Augment a batch of data
        
        Args:
            X: Input data [n_samples, n_channels, n_times]
            y: Labels [n_samples]
            augmentation_factor: Number of augmented samples to generate per original sample
            
        Returns:
            X_aug: Augmented data [n_samples * (augmentation_factor + 1), n_channels, n_times]
            y_aug: Augmented labels [n_samples * (augmentation_factor + 1)]
        """
        n_samples = X.shape[0]
        
        # Store original samples for mixup
        self.trial_cache = {
            'data': X.copy(),
            'labels': y.copy() if y is not None else None
        }
        
        # Initialize augmented data arrays
        X_aug = np.zeros((n_samples * (augmentation_factor + 1),) + X.shape[1:], dtype=X.dtype)
        y_aug = np.zeros(n_samples * (augmentation_factor + 1), dtype=int if y is not None else float)
        
        # Copy original samples
        X_aug[:n_samples] = X
        if y is not None:
            y_aug[:n_samples] = y
        
        # Generate augmented samples
        for i in range(n_samples):
            for j in range(augmentation_factor):
                aug_idx = n_samples + i * augmentation_factor + j
                X_aug[aug_idx] = self.augment(X[i], y[i] if y is not None else None)
                if y is not None:
                    y_aug[aug_idx] = y[i]
        
        return X_aug, y_aug


class BasicAugmenter(EEGDataAugmenter):
    """
    Basic augmentation strategy using only simple methods
    """
    def __init__(self, noise_level=0.03, scale_range=(0.9, 1.1), chunk_max_offset=30):
        super().__init__(noise_level, scale_range, chunk_max_offset, support_variable_length=False)
        
        # Just basic methods
        self.basic_methods = {
            'add_noise': self.add_noise,
            'scale_amplitude': self.scale_amplitude
        }
    
    def augment(self, x, y=None):
        """Apply basic augmentation strategy"""
        x_aug = x.copy()
        
        # Get basic augmentation methods
        augmentations = list(self.basic_methods.values())
        
        # Apply 1-2 augmentations
        num_augmentations = np.random.randint(1, 3)
        selected_augmentations = np.random.choice(augmentations, 
                                                size=min(num_augmentations, len(augmentations)), 
                                                replace=False)
        
        # Apply selected augmentations
        for augmentation in selected_augmentations:
            x_aug = augmentation(x_aug)
            
        return x_aug
    
    def augment_batch(self, X, y, augmentation_factor=2):
        """
        Augment a batch of data
        
        Args:
            X: Input data [n_samples, n_channels, n_times]
            y: Labels [n_samples]
            augmentation_factor: Number of augmented samples to generate per original sample
            
        Returns:
            X_aug: Augmented data [n_samples * (augmentation_factor + 1), n_channels, n_times]
            y_aug: Augmented labels [n_samples * (augmentation_factor + 1)]
        """
        n_samples = X.shape[0]
        
        # Store original samples for potential mixup
        self.trial_cache = {
            'data': X.copy(),
            'labels': y.copy() if y is not None else None
        }
        
        # Initialize augmented data arrays
        X_aug = np.zeros((n_samples * (augmentation_factor + 1),) + X.shape[1:], dtype=X.dtype)
        y_aug = np.zeros(n_samples * (augmentation_factor + 1), dtype=int if y is not None else float)
        
        # Copy original samples
        X_aug[:n_samples] = X
        if y is not None:
            y_aug[:n_samples] = y
        
        # Generate augmented samples
        for i in range(n_samples):
            for j in range(augmentation_factor):
                aug_idx = n_samples + i * augmentation_factor + j
                X_aug[aug_idx] = self.augment(X[i], y[i] if y is not None else None)
                if y is not None:
                    y_aug[aug_idx] = y[i]
        
        return X_aug, y_aug


if __name__ == "__main__":
    import time
    import os.path
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Step 0: Split subjects into training and test sets
    # Reserve 10 subjects for testing/fine-tuning
    train_subjects, test_subjects = split_subjects(total_subjects=109, test_subjects_count=10, random_seed=42)
    
    # Print subject split information
    print("Training subjects:", train_subjects[:5], "...", f"(total: {len(train_subjects)})")
    print("Test subjects:", test_subjects)
    
    # Define the model path
    model_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(model_dir, exist_ok=True)
    base_model_name = "base_model_optimized"
    
    # Check if we need to train the base model
    need_to_train_base = True
    # Look for existing model files
    for n_subjects in range(70, 100):  # Try different subject counts
        potential_model_path = os.path.join(model_dir, f'{base_model_name}_{n_subjects}_subjects.pt')
        if os.path.exists(potential_model_path):
            base_model_path = potential_model_path
            print(f"Found existing base model: {base_model_path}")
            need_to_train_base = False
            break
    
    # Step 1: Train base model if needed
    if need_to_train_base:
        print("No existing base model found. Training new base model...")
        base_classifier, base_results, metadata = train_base_model_multi_subject(
            run_ids=RUN_IDS,
            n_subjects=80,      # Use 80 subjects for pretraining
            subject_ids=train_subjects,  # Only use training subjects
            save_model=True,
            n_epochs=200,       # Train for longer since it's pretraining
            model_name=base_model_name
        )
        base_model_path = os.path.join(model_dir, f'{base_model_name}_{metadata["n_subjects"]}_subjects.pt')
    else:
        print(f"Using existing base model: {base_model_path}")
    
    # Step 2: Fine-tune for individual test subjects
    fine_tune_results = {}
    for subject in test_subjects[0:1]:
        try:
            # Fine-tune for this test subject
            print(f"\nFine-tuning for subject {subject}...")
            subject_results = finetune_subject_specific(
                base_model_path=base_model_path,
                subject_id=subject,
                run_ids=RUN_IDS,
                n_epochs=100,
                save_models=True
            )
            fine_tune_results[subject] = subject_results
        except Exception as e:
            print(f"Error fine-tuning subject {subject}: {str(e)}")
            print("Continuing with next subject...")
            continue
    
    plt.show()
