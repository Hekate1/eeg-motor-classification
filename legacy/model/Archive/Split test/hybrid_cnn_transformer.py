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
import math
import traceback  # Add traceback module for better error handling
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

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
    def __init__(self, n_channels, n_times, embedding_dim=128, dropout=0.2):
        super(EEGCNNEncoder, self).__init__()
        
        # First block: Temporal convolution followed by spatial convolution
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 64), padding='same'),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=(n_channels, 1), padding=0),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout)
        )
        
        # Second block: Separable convolution
        self.block2 = nn.Sequential(
            DepthwiseSeparableConv(32, 64, kernel_size=(1, 16)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout)
        )
        
        # Calculate output size
        self.n_times_out = n_times // 32  # After two pooling layers
        
        # Final convolution to reduce to embedding dimension
        self.final_conv = nn.Conv2d(64, embedding_dim, kernel_size=1)
        self.flat_size = embedding_dim * self.n_times_out
        
    def forward(self, x):
        # x shape: [batch_size, n_channels, n_times]
        batch_size = x.size(0)
        
        # Add channel dimension for 2D convolution
        x = x.unsqueeze(1)  # [batch_size, 1, n_channels, n_times]
        
        # Apply convolutional blocks
        x = self.block1(x)
        x = self.block2(x)
        
        # Final convolution
        x = self.final_conv(x)
        
        # Get actual output time dimension
        _, channels, _, time_out = x.shape
        
        # Reshape for transformer: [batch_size, n_times_out, embedding_dim]
        x = x.squeeze(2).permute(0, 2, 1)
        
        return x


class HybridCNNTransformer(nn.Module):
    """
    Hybrid CNN/Transformer model for EEG classification
    with CSP and frequency feature integration
    """
    def __init__(self, n_channels, n_times, n_classes, 
                 n_csp_components=4, n_freq_features=None,
                 embedding_dim=128, n_heads=4, n_layers=2, 
                 dropout=0.2, use_csp=True, use_freq=True):
        super(HybridCNNTransformer, self).__init__()
        
        self.use_csp = use_csp
        self.use_freq = use_freq
        
        # CNN encoder for raw EEG
        self.cnn_encoder = EEGCNNEncoder(
            n_channels=n_channels,
            n_times=n_times,
            embedding_dim=embedding_dim,
            dropout=dropout
        )
        
        # Feature size calculation
        self.n_times_out = n_times // 32  # After CNN pooling
        
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
        self.seq_len = self.n_times_out
        if use_csp:
            self.seq_len += 1  # Add one token for CSP features
        if use_freq and n_freq_features is not None:
            self.seq_len += 1  # Add one token for frequency features
        
        # Store key parameters
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        self.n_freq_features = n_freq_features
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=embedding_dim,
            max_len=self.seq_len + 10,  # Add some buffer
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
        
        # Special initialization for the classifier to avoid initial bias
        # This helps prevent the "predict all one class" problem
        if hasattr(self, 'classifier') and isinstance(self.classifier, nn.Sequential):
            # Find the final layer in the classifier
            final_layer = None
            for module in self.classifier:
                if isinstance(module, nn.Linear):
                    final_layer = module
            
            if final_layer is not None:
                # Use a stronger initialization to help the model learn class differences
                nn.init.xavier_normal_(final_layer.weight, gain=2.0)  # Increased gain for stronger gradients
                
                # Initialize the bias to explicitly favor balanced predictions
                if hasattr(final_layer, 'bias') and final_layer.bias is not None:
                    # Explicitly set a slight initial preference for different classes
                    n_classes = final_layer.bias.shape[0]
                    if n_classes == 2:  # Special case for binary classification
                        # Set slight opposite biases for the two classes
                        final_layer.bias.data[0] = 0.1
                        final_layer.bias.data[1] = -0.1
                    else:
                        # For multi-class, alternate between positive and negative small biases
                        for i in range(n_classes):
                            final_layer.bias.data[i] = 0.1 * (1 if i % 2 == 0 else -1)
    
    def forward(self, x_raw, x_csp=None, x_freq=None, attention_mask=None, force_balanced=False):
        """Forward pass through the model.
        
        Parameters:
        -----------
        x_raw : tensor
            Raw EEG data
        x_csp : tensor, optional
            CSP features
        x_freq : tensor, optional
            Frequency features
        attention_mask : tensor, optional
            Attention mask for variable length sequences
        force_balanced : bool, optional
            If True, forces more balanced predictions when the model might be stuck
            
        Returns:
        --------
        tensor
            Output class logits
        """
        # Run standard forward pass
        if attention_mask is not None:
            batch_size = x_raw.size(0)
            
            # Encode raw EEG with CNN
            cnn_features = self.cnn_encoder(x_raw)  # [batch_size, n_times_out, embedding_dim]
            
            # Fix sequence length mismatch by truncating or padding
            _, cnn_time_dim, _ = cnn_features.shape
            if cnn_time_dim != self.n_times_out:
                print(f"WARNING: CNN output time dimension {cnn_time_dim} doesn't match expected {self.n_times_out}")
                if cnn_time_dim > self.n_times_out:
                    # Truncate extra timesteps
                    print(f"Runtime adapter: Truncating CNN output from {cnn_time_dim} to {self.n_times_out}")
                    cnn_features = cnn_features[:, :self.n_times_out, :]
                else:
                    # Pad missing timesteps
                    print(f"Runtime adapter: Padding CNN output from {cnn_time_dim} to {self.n_times_out}")
                    padding = torch.zeros(batch_size, self.n_times_out - cnn_time_dim, 
                                         self.embedding_dim, device=cnn_features.device)
                    cnn_features = torch.cat([cnn_features, padding], dim=1)
            
            # Apply the attention mask to CNN features (matching the sequence length)
            # We need to downsample the mask to match CNN output
            cnn_mask = attention_mask[:, ::32]  # Adjust based on your actual pooling
            cnn_mask = cnn_mask[:, :cnn_features.size(1)]  # Make sure lengths match
            
            # Prepare sequence for transformer
            sequence = [cnn_features]
            
            # Create a mask for the full sequence
            full_mask = cnn_mask.clone()
            
            # Add CSP features if available
            if self.use_csp and x_csp is not None:
                # Encode CSP features to embedding dimension
                csp_features = self.csp_encoder(x_csp)  # [batch_size, embedding_dim]
                # Add as a separate token
                csp_token = csp_features.unsqueeze(1)  # [batch_size, 1, embedding_dim]
                sequence.append(csp_token)
                
                # Add mask entries for CSP tokens (always attend to these)
                csp_token_mask = torch.ones(batch_size, 1, device=cnn_mask.device)
                full_mask = torch.cat([full_mask, csp_token_mask], dim=1)
            
            # Add frequency features if available
            if self.use_freq and x_freq is not None:
                # Encode frequency features to embedding dimension
                freq_features = self.freq_encoder(x_freq)  # [batch_size, embedding_dim]
                # Add as a separate token
                freq_token = freq_features.unsqueeze(1)  # [batch_size, 1, embedding_dim]
                sequence.append(freq_token)
                
                # Add mask entries for frequency tokens (always attend to these)
                freq_token_mask = torch.ones(batch_size, 1, device=cnn_mask.device)
                full_mask = torch.cat([full_mask, freq_token_mask], dim=1)
            
            # Concatenate all features into a single sequence
            # Shape: [batch_size, seq_len, embedding_dim]
            sequence = torch.cat(sequence, dim=1)
            
            # Apply positional encoding
            sequence = self.pos_encoder(sequence)
            
            # Create attention mask for transformer (convert to 2D mask)
            # Shape: [batch_size, seq_len, seq_len]
            attn_mask = full_mask.unsqueeze(1).repeat(1, full_mask.size(1), 1)
            
            # Apply transformer layers with attention mask
            for layer in self.transformer_layers:
                sequence = layer(sequence, attn_mask)
            
            # Global average pooling across time dimension
            # Use the mask to only average over valid positions
            mask_expanded = full_mask.unsqueeze(-1).expand_as(sequence)
            sequence = sequence * mask_expanded
            pooled = sequence.sum(dim=1) / full_mask.sum(dim=1, keepdim=True)
            
            # Apply layer normalization
            normalized = self.layer_norm(pooled)
            
            # Apply classification head
            logits = self.classifier(normalized)
            
            # Apply balancing adjustment if requested (for validation mode)
            if force_balanced and not self.training:
                # Get class with highest average prediction
                with torch.no_grad():
                    probs = torch.softmax(logits, dim=1)
                    class_means = probs.mean(dim=0)
                    max_class = torch.argmax(class_means)
                    
                    # Only apply adjustment if predictions are very unbalanced
                    if class_means[max_class] > 0.75:  # If one class dominates
                        # Apply a calibration factor to logits to balance predictions
                        adjustment = torch.zeros_like(logits)
                        adjustment[:, max_class] = -1.0  # Penalize dominant class
                        for c in range(self.n_classes):
                            if c != max_class:
                                # Boost other classes
                                adjustment[:, c] = 1.0 / (self.n_classes - 1)
                        
                        # Apply adjustment
                        logits = logits + adjustment
            
            return logits
            
        else:
            # Original forward pass without variable length handling
            batch_size = x_raw.size(0)
            
            # Encode raw EEG with CNN
            cnn_features = self.cnn_encoder(x_raw)  # [batch_size, n_times_out, embedding_dim]
            
            # Fix sequence length mismatch by truncating or padding
            _, cnn_time_dim, _ = cnn_features.shape
            if cnn_time_dim != self.n_times_out:
                print(f"WARNING: CNN output time dimension {cnn_time_dim} doesn't match expected {self.n_times_out}")
                if cnn_time_dim > self.n_times_out:
                    # Truncate extra timesteps
                    print(f"Runtime adapter: Truncating CNN output from {cnn_time_dim} to {self.n_times_out}")
                    cnn_features = cnn_features[:, :self.n_times_out, :]
                else:
                    # Pad missing timesteps
                    print(f"Runtime adapter: Padding CNN output from {cnn_time_dim} to {self.n_times_out}")
                    padding = torch.zeros(batch_size, self.n_times_out - cnn_time_dim, 
                                         self.embedding_dim, device=cnn_features.device)
                    cnn_features = torch.cat([cnn_features, padding], dim=1)
            
            # Prepare sequence for transformer
            sequence = [cnn_features]
            
            # Add CSP features if available
            if self.use_csp and x_csp is not None:
                # Encode CSP features to embedding dimension
                csp_features = self.csp_encoder(x_csp)  # [batch_size, embedding_dim]
                # Add as a separate token
                csp_token = csp_features.unsqueeze(1)  # [batch_size, 1, embedding_dim]
                sequence.append(csp_token)
            
            # Add frequency features if available
            if self.use_freq and x_freq is not None:
                # Encode frequency features to embedding dimension
                freq_features = self.freq_encoder(x_freq)  # [batch_size, embedding_dim]
                # Add as a separate token
                freq_token = freq_features.unsqueeze(1)  # [batch_size, 1, embedding_dim]
                sequence.append(freq_token)
            
            # Concatenate all features into a single sequence
            # Shape: [batch_size, seq_len, embedding_dim]
            sequence = torch.cat(sequence, dim=1)
            
            # Verify final sequence length matches expected model sequence length
            actual_seq_len = sequence.size(1)
            if actual_seq_len != self.seq_len:
                # print(f"WARNING: Final sequence length {actual_seq_len} doesn't match model's expected length {self.seq_len}")
                if actual_seq_len > self.seq_len:
                    # Truncate if too long
                    # print(f"Runtime adapter: Truncating final sequence from {actual_seq_len} to {self.seq_len}")
                    sequence = sequence[:, :self.seq_len, :]
                else:
                    # Pad if too short
                    print(f"Runtime adapter: Padding final sequence from {actual_seq_len} to {self.seq_len}")
                    padding = torch.zeros(batch_size, self.seq_len - actual_seq_len, 
                                         self.embedding_dim, device=sequence.device)
                    sequence = torch.cat([sequence, padding], dim=1)
            
            # Apply positional encoding
            sequence = self.pos_encoder(sequence)
            
            # Apply transformer layers
            for layer in self.transformer_layers:
                sequence = layer(sequence)
            
            # Global average pooling across time dimension
            pooled = torch.mean(sequence, dim=1)  # [batch_size, embedding_dim]
            
            # Apply layer normalization
            normalized = self.layer_norm(pooled)
            
            # Apply classification head
            logits = self.classifier(normalized)
            
            # Apply balancing adjustment if requested (for validation mode)
            if force_balanced and not self.training:
                # Get class with highest average prediction
                with torch.no_grad():
                    probs = torch.softmax(logits, dim=1)
                    class_means = probs.mean(dim=0)
                    max_class = torch.argmax(class_means)
                    
                    # Only apply adjustment if predictions are very unbalanced
                    if class_means[max_class] > 0.75:  # If one class dominates
                        # Apply a calibration factor to logits to balance predictions
                        adjustment = torch.zeros_like(logits)
                        adjustment[:, max_class] = -1.0  # Penalize dominant class
                        for c in range(self.n_classes):
                            if c != max_class:
                                # Boost other classes
                                adjustment[:, c] = 1.0 / (self.n_classes - 1)
                        
                        # Apply adjustment
                        logits = logits + adjustment
            
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
    
    def augment_batch(self, X, y, augmentation_factor=3):
        """Default implementation that just returns original data"""
        print("WARNING: Using default augment_batch implementation that returns original data")
        return X, y


class HybridModelClassifier:
    """
    Wrapper class for training and evaluating the Hybrid CNN/Transformer model
    """
    def __init__(self, n_classes, embedding_dim=128, n_heads=4, n_layers=2,
                 dropout=0.3, lr=0.0005, batch_size=32, n_epochs=200,
                 weight_decay=0.01, device=None, augmentation_factor=2,
                 use_csp=True, use_freq=True, n_csp_components=4,
                 variable_length_augmentation=True, force_balanced_val=False):
        
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
        self.force_balanced_val = force_balanced_val  # New flag for balanced validation
        
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
            
        save_dict = {
            'model_state_dict': self.model.state_dict(),
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
            'train_config': {
                'lr': self.lr,
                'batch_size': self.batch_size,
                'n_epochs': self.n_epochs,
                'weight_decay': self.weight_decay,
                'augmentation_factor': self.augmentation_factor,
                'variable_length_augmentation': self.variable_length_augmentation
            },
            'metadata': metadata  # Add the metadata dictionary
        }
        
        torch.save(save_dict, filepath)
    
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
        print(f"Using device: {device}")
        
        try:
            save_dict = torch.load(filepath, map_location=device)
            
            # Check for old format vs new format
            if 'model_config' in save_dict:
                model_config = save_dict['model_config']
                train_config = save_dict['train_config']
            else:
                # Handle older model format
                model_config = save_dict['config']
                train_config = {
                    'lr': 0.0005,
                    'batch_size': 32,
                    'n_epochs': 150,
                    'weight_decay': 0.01,
                    'augmentation_factor': 2,
                    'variable_length_augmentation': False
                }
                
            metadata = save_dict.get('metadata', {})  # Get metadata if it exists
            
            # Create model instance
            classifier = cls(
                n_classes=model_config['n_classes'],
                embedding_dim=model_config['embedding_dim'],
                n_heads=model_config['n_heads'],
                n_layers=model_config['n_layers'],
                dropout=model_config['dropout'],
                lr=train_config.get('lr', 0.0005),
                batch_size=train_config.get('batch_size', 32),
                n_epochs=train_config.get('n_epochs', 150),
                weight_decay=train_config.get('weight_decay', 0.01),
                device=device,
                augmentation_factor=train_config.get('augmentation_factor', 2),
                use_csp=model_config['use_csp'],
                use_freq=model_config['use_freq'],
                n_csp_components=model_config['n_csp_components'],
                variable_length_augmentation=train_config.get('variable_length_augmentation', False)
            )
            
            # Extract dimensions from the state_dict to ensure compatibility
            state_dict = save_dict['model_state_dict']
            
            # Get n_channels from the first convolution layer weight
            n_channels = None
            if 'cnn_encoder.block1.2.weight' in state_dict:
                # The CNN layer weight shape is [out_channels, in_channels, n_channels, kernel_size]
                _, _, n_channels, _ = state_dict['cnn_encoder.block1.2.weight'].shape
            
            if n_channels is None:
                n_channels = 64  # Default fallback
                print(f"Could not determine n_channels from state_dict, using default: {n_channels}")
            else:
                print(f"Extracted n_channels from state_dict: {n_channels}")
            
            # Extract sequence length directly from the positional encoding
            seq_len = None
            if 'pos_encoder.pe' in state_dict:
                # Extract sequence length from positional encoding
                _, seq_len, _ = state_dict['pos_encoder.pe'].shape
                print(f"Extracted sequence length from positional encoding: {seq_len}")
            
            # Get n_times from sequence length or use default
            n_times = 640  # Default
            if seq_len is not None:
                # Estimate n_times by multiplying seq_len by 32 (typical CNN downsampling factor)
                n_times = seq_len * 32
                print(f"Estimated n_times from sequence length: {n_times}")
            
            # Get n_freq_features from the frequency encoder if available
            n_freq_features = None
            if model_config['use_freq'] and 'freq_encoder.0.weight' in state_dict:
                # The frequency encoder input size is the second dimension of the weight
                _, n_freq_features = state_dict['freq_encoder.0.weight'].shape
                print(f"Extracted n_freq_features from state_dict: {n_freq_features}")
            
            # IMPORTANT: Force the seq_len to match exactly between models
            # This fixes the positional encoding shape mismatch issue
            if seq_len is not None:
                # We need to modify the CNN output calculation to ensure seq_len matches
                # Calculate CNN output size based on n_times that would produce exactly the desired seq_len
                cnn_output_size = seq_len
                # Recreate n_times if needed to ensure consistent shape
                n_times = seq_len * 32
            else:
            # Calculate actual CNN output size based on n_times 
                cnn_output_size = n_times // 32
            
            # Create the model with the exact dimensions from the checkpoint
            classifier.model = HybridCNNTransformer(
                n_channels=n_channels,
                n_times=n_times,
                n_classes=model_config['n_classes'],
                n_csp_components=model_config['n_csp_components'],
                n_freq_features=n_freq_features,
                embedding_dim=model_config['embedding_dim'],
                n_heads=model_config['n_heads'],
                n_layers=model_config['n_layers'],
                dropout=model_config['dropout'],
                use_csp=model_config['use_csp'],
                use_freq=model_config['use_freq']
            ).to(device)
            
            # IMPORTANT: Make sure the seq_len in the model matches exactly what's in the state dict
            if seq_len is not None and hasattr(classifier.model, 'seq_len'):
                # Force the model's sequence length to match the saved model
                classifier.model.seq_len = seq_len
                print(f"Forced model sequence length to match saved model: {seq_len}")
                
                # Manually modify the positional encoding to have the exact same shape before loading weights
                if hasattr(classifier.model, 'pos_encoder') and hasattr(classifier.model.pos_encoder, 'pe'):
                    # Get embedding dimension
                    embedding_dim = model_config['embedding_dim']
                    # Recreate positional encoding with the EXACT same shape as in the state dict
                    pe = torch.zeros(1, seq_len, embedding_dim, device=device)
                    position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
                    div_term = torch.exp(torch.arange(0, embedding_dim, 2, device=device).float() * (-np.log(10000.0) / embedding_dim))
                    pe[0, :, 0::2] = torch.sin(position * div_term)
                    pe[0, :, 1::2] = torch.cos(position * div_term)
                    # Directly assign to ensure exact shape match
                    classifier.model.pos_encoder.pe = pe
                    print(f"Directly set positional encoding shape to match saved model: {pe.shape}")
            
            # Load the state dict
            try:
                classifier.model.load_state_dict(state_dict, strict=True)
                print("Successfully loaded model state dictionary with strict=True")
            except Exception as e:
                print(f"Strict loading failed: {str(e)}")
                
                # If the error is due to positional encoding mismatch
                if "size mismatch for pos_encoder.pe" in str(e):
                    print("Detected positional encoding size mismatch. Attempting to fix...")
                    
                    # Get expected position encoding shape from state dict
                    if 'pos_encoder.pe' in state_dict:
                        expected_pe_shape = state_dict['pos_encoder.pe'].shape
                        print(f"Expected positional encoding shape from state dict: {expected_pe_shape}")
                        
                        # Create new positional encoding with exact same shape
                        pe = torch.zeros(expected_pe_shape, device=device)
                        _, seq_len, d_model = expected_pe_shape
                        
                        # Compute values
                        position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
                        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-np.log(10000.0) / d_model))
                        pe[0, :, 0::2] = torch.sin(position * div_term)
                        pe[0, :, 1::2] = torch.cos(position * div_term)
                        
                        # Replace the position encoding in the model
                        classifier.model.pos_encoder.pe = pe
                        print(f"Replaced positional encoding with shape {pe.shape}")
                        
                        # Now force the model's seq_len to match
                        classifier.model.seq_len = seq_len
                        print(f"Set model sequence length to {seq_len}")
                        
                        # Try loading again
                        try:
                            # Remove problematic positional encoding key from state dict
                            state_dict.pop('pos_encoder.pe', None)
                            classifier.model.load_state_dict(state_dict, strict=False)
                            print("Successfully loaded model after fixing positional encoding")
                        except Exception as e2:
                            print(f"Still failed to load model: {str(e2)}")
                            print("Attempting to load with strict=False...")
                            classifier.model.load_state_dict(state_dict, strict=False)
                            print("Successfully loaded model state dictionary with strict=False")
                else:
                print("Attempting to load with strict=False...")
                # Fall back to non-strict loading if strict loading fails
                classifier.model.load_state_dict(state_dict, strict=False)
                print("Successfully loaded model state dictionary with strict=False")
            
            print(f"Successfully loaded model from {filepath}")
            return classifier, metadata
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise e
    
    def prepare_data(self, X_raw, y, X_csp=None, X_freq=None, val_split=0.2, apply_augmentation=True, 
                     train_indices=None, val_indices=None):
        """
        Prepare data for training and validation.
        
        Args:
            X_raw: Raw EEG data [n_trials, n_channels, n_times]
            y: Labels [n_trials]
            X_csp: CSP features [n_trials, n_components]
            X_freq: Frequency features [n_trials, n_features]
            val_split: Validation split ratio
            apply_augmentation: Whether to apply data augmentation
            train_indices: Indices for training set (if None, will be created based on val_split)
            val_indices: Indices for validation set (if None, will be created based on val_split)
            
        Returns:
            train_loader: DataLoader for training set
            val_loader: DataLoader for validation set
        """
        device = self.device
        n_trials = X_raw.shape[0]
        
        # Convert to tensors if needed
        if not isinstance(X_raw, torch.Tensor):
            X_raw = torch.FloatTensor(X_raw)
        if not isinstance(y, torch.Tensor):
            y = torch.LongTensor(y)
            
        # Move to device
        X_raw = X_raw.to(device)
        y = y.to(device)
        
        if X_csp is not None and not isinstance(X_csp, torch.Tensor):
            X_csp = torch.FloatTensor(X_csp).to(device)
            
        if X_freq is not None and not isinstance(X_freq, torch.Tensor):
            X_freq = torch.FloatTensor(X_freq).to(device)
        
        # Use provided indices if available, otherwise create split
        if train_indices is None or val_indices is None:
            # Create indices for stratified split
            indices_by_class = [np.where(y.cpu().numpy() == i)[0] for i in range(self.n_classes)]
            
            train_indices = []
            val_indices = []
            
            for class_indices in indices_by_class:
                np.random.shuffle(class_indices)
                val_size = int(len(class_indices) * val_split)
                
                val_indices.extend(class_indices[:val_size])
                train_indices.extend(class_indices[val_size:])
        
            # Convert to numpy arrays
            train_indices = np.array(train_indices)
            val_indices = np.array(val_indices)
            
        # Select training and validation data
        X_raw_train = X_raw[train_indices]
        y_train = y[train_indices]
        X_raw_val = X_raw[val_indices]
        y_val = y[val_indices]
        
        # Select CSP features for training and validation (if available)
        X_csp_train = X_csp[train_indices] if X_csp is not None else None
        X_csp_val = X_csp[val_indices] if X_csp is not None else None
        
        # Select frequency features for training and validation (if available)
        X_freq_train = X_freq[train_indices] if X_freq is not None else None
        X_freq_val = X_freq[val_indices] if X_freq is not None else None
        
        # Store the original training data before augmentation
        X_raw_train_orig = X_raw_train.clone()
        y_train_orig = y_train.clone()
        
        # Apply data augmentation to training set if requested
        if apply_augmentation and self.augmenter is not None:
            try:
                print(f"Applying data augmentation with {type(self.augmenter).__name__}")
                
                # Extra check to ensure the augmenter has the augment_batch method
                if not hasattr(self.augmenter, 'augment_batch'):
                    print("WARNING: Augmenter doesn't have augment_batch method. Using original data without augmentation.")
                    # Convert to numpy for consistency
                    X_raw_train_np = X_raw_train.cpu().numpy()
                    y_train_np = y_train.cpu().numpy()
                    X_aug, y_aug = X_raw_train_np, y_train_np
            else:
                    # Debug: Show shapes before augmentation
                    print(f"Before augmentation - X_raw_train: {X_raw_train.shape}, y_train: {y_train.shape}")
                    
                    # Convert to numpy for augmentation
                    X_raw_train_np = X_raw_train.cpu().numpy() 
                    y_train_np = y_train.cpu().numpy()
                    
                    # Apply augmentation with explicit shape checking
                    X_aug, y_aug = self.augmenter.augment_batch(X_raw_train_np, y_train_np, 
                                                            augmentation_factor=self.augmentation_factor)
                
                # Debug: Show shapes after augmentation
                print(f"After augmentation - X_aug: {X_aug.shape}, y_aug: {y_aug.shape}")
                
                # Verify shapes match
                if len(X_aug) != len(y_aug):
                    print(f"WARNING: Augmentation produced mismatched shapes: X({len(X_aug)}) vs y({len(y_aug)})")
                    print("Falling back to original data")
                    # Use original data if augmentation failed
                    X_aug, y_aug = X_raw_train_np, y_train_np
                
                # Convert back to tensors
                X_raw_train = torch.FloatTensor(X_aug).to(device)
                y_train = torch.LongTensor(y_aug).to(device)
                
                print(f"Augmented training set: {len(y_train)} samples")
                print(f"Final tensor shapes - X_raw_train: {X_raw_train.shape}, y_train: {y_train.shape}")
                
                # Recompute CSP and frequency features for the augmented data if needed
                if self.use_csp and X_csp is not None:
                    try:
                        # Only extract new features if we have enough samples of each class
                        # Clone to avoid modifying original data during extraction
                        X_aug_cpu = X_aug.copy()
                        y_aug_cpu = y_aug.copy()
                        
                        # Check if we have at least 2 samples per class (minimum for CSP)
                        class_counts = np.bincount(y_aug_cpu, minlength=self.n_classes)
                        if np.min(class_counts) >= 2:
                            print("Recalculating CSP features for augmented data...")
                            # Extract CSP features from augmented data
                            try:
                                # We need to re-use the original CSP transformer to maintain consistency
                                if self.csp_transformer is None:
                                    # First time, extract CSP features and store the transformer
                                    X_csp_aug, self.csp_transformer = extract_csp_features(
                                        X_aug_cpu, y_aug_cpu, n_components=self.n_csp_components
                                    )
                                else:
                                    # Apply the stored transformer to get consistent features
                                    X_aug_shaped = X_aug_cpu.reshape(X_aug_cpu.shape[0], -1)
                                    X_csp_aug = self.csp_transformer.transform(X_aug_shaped)
                                
                                # Convert to tensor
                                X_csp_train = torch.FloatTensor(X_csp_aug).to(device)
                                print(f"Successfully recalculated CSP features: {X_csp_train.shape}")
                            except Exception as e:
                                print(f"Error recalculating CSP features: {str(e)}")
                                print("Falling back to duplicating CSP features")
                                # Will fall back to duplication later
                        else:
                            print(f"Not enough samples per class for CSP: {class_counts}")
                            print("Falling back to duplicating CSP features")
                            # Will fall back to duplication later
                    except Exception as e:
                        print(f"Error in CSP feature recalculation logic: {str(e)}")
                        # Will fall back to duplication later
                
                if self.use_freq and X_freq is not None:
                    try:
                        print("Recalculating frequency features for augmented data...")
                        # Extract frequency features from augmented data
                        X_freq_aug = extract_frequency_features(
                            X_aug, bands=[(8, 12), (13, 30)]  # Mu and beta bands
                        )
                        
                        # Convert to tensor
                        X_freq_train = torch.FloatTensor(X_freq_aug).to(device)
                        print(f"Successfully recalculated frequency features: {X_freq_train.shape}")
                    except Exception as e:
                        print(f"Error recalculating frequency features: {str(e)}")
                        print("Falling back to duplicating frequency features")
                        # Will fall back to duplication later
            except Exception as e:
                print(f"Error during data augmentation: {str(e)}")
                print(f"Traceback: {traceback.format_exc()}")
                print("Using original data without augmentation")
                # No change to X_raw_train, y_train
            
        # Normalize the data
        X_raw_train = subject_specific_normalize(X_raw_train.cpu().numpy())
        X_raw_val = subject_specific_normalize(X_raw_val.cpu().numpy())
        
        # Convert back to tensors (normalization returned numpy arrays)
        X_raw_train = torch.FloatTensor(X_raw_train).to(device)
        X_raw_val = torch.FloatTensor(X_raw_val).to(device)
            
        if self.use_csp and X_csp_train is not None:
            # If using CSP, possibly needs reshaping to match augmented data
            if len(X_csp_train) != len(X_raw_train):
                print(f"WARNING: X_csp_train size ({len(X_csp_train)}) doesn't match X_raw_train ({len(X_raw_train)})")
                print("Duplicating CSP features to match augmented data")
                # Find the original training samples in the augmented set
                # (usually the first n_original samples are the original ones)
                n_original = len(train_indices)
                n_augmented = len(X_raw_train) - n_original
                
                # Duplicate the CSP features for augmented data
                if n_augmented > 0:
                    # Create augmented CSP by repeating original CSP features
                    augmentation_factor = len(X_raw_train) // len(X_csp_train)
                    X_csp_train = X_csp_train.repeat(augmentation_factor, 1)
                    
                    # Ensure shapes match exactly
                    if len(X_csp_train) > len(X_raw_train):
                        X_csp_train = X_csp_train[:len(X_raw_train)]
                    elif len(X_csp_train) < len(X_raw_train):
                        # Fill missing with zeros
                        padding = torch.zeros(len(X_raw_train) - len(X_csp_train), 
                                             X_csp_train.shape[1], device=device)
                        X_csp_train = torch.cat([X_csp_train, padding], dim=0)
                        
            print(f"Final X_csp_train shape: {X_csp_train.shape}")
            
        if self.use_freq and X_freq_train is not None:
            # If using frequency features, possibly needs reshaping to match augmented data
            if len(X_freq_train) != len(X_raw_train):
                print(f"WARNING: X_freq_train size ({len(X_freq_train)}) doesn't match X_raw_train ({len(X_raw_train)})")
                print("Duplicating frequency features to match augmented data")
                # Duplicate the frequency features for augmented data
                augmentation_factor = len(X_raw_train) // len(X_freq_train)
                X_freq_train = X_freq_train.repeat(augmentation_factor, 1)
                
                # Ensure shapes match exactly
                if len(X_freq_train) > len(X_raw_train):
                    X_freq_train = X_freq_train[:len(X_raw_train)]
                elif len(X_freq_train) < len(X_raw_train):
                    # Fill missing with zeros
                    padding = torch.zeros(len(X_raw_train) - len(X_freq_train), 
                                         X_freq_train.shape[1], device=device)
                    X_freq_train = torch.cat([X_freq_train, padding], dim=0)
                    
            print(f"Final X_freq_train shape: {X_freq_train.shape}")
        
        # Create training dataset
        if self.use_csp and self.use_freq and X_csp_train is not None and X_freq_train is not None:
            train_dataset = TensorDataset(X_raw_train, X_csp_train, X_freq_train, y_train)
        elif self.use_csp and X_csp_train is not None:
            train_dataset = TensorDataset(X_raw_train, X_csp_train, y_train)
        elif self.use_freq and X_freq_train is not None:
            train_dataset = TensorDataset(X_raw_train, X_freq_train, y_train)
            else:
            train_dataset = TensorDataset(X_raw_train, y_train)
        
        # Create validation dataset
        if self.use_csp and self.use_freq and X_csp_val is not None and X_freq_val is not None:
            val_dataset = TensorDataset(X_raw_val, X_csp_val, X_freq_val, y_val)
        elif self.use_csp and X_csp_val is not None:
            val_dataset = TensorDataset(X_raw_val, X_csp_val, y_val)
        elif self.use_freq and X_freq_val is not None:
            val_dataset = TensorDataset(X_raw_val, X_freq_val, y_val)
        else:
            val_dataset = TensorDataset(X_raw_val, y_val)
        
        # Create data loaders
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

    def train_and_evaluate(self, X_raw, y, val_split=0.2, early_stopping_patience=30, custom_class_weights=None):
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
        
        # Check if model has already been initialized (during loading)
        if self.model is None:
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
        else:
            # Model was already initialized (likely from loading)
            # Check if dimensions are compatible with our data
            if hasattr(self.model, 'n_channels') and self.model.n_channels != n_channels:
                print(f"Warning: Model expects {self.model.n_channels} channels but data has {n_channels} channels")
                # This is a critical error that must be fixed
                raise ValueError(f"Channel count mismatch: model={self.model.n_channels}, data={n_channels}")
            
            # For n_times, we're more flexible since the CNN will downsample
            # The critical thing is the CNN output size (seq_len) must match what's expected
            if hasattr(self.model, 'seq_len'):
                model_seq_len = self.model.seq_len
                current_seq_len = n_times // 32  # Expected sequence length from current data
                
                if current_seq_len != model_seq_len:
                    print(f"Warning: Data would produce sequence length {current_seq_len} but model expects {model_seq_len}")
                    print("This would cause dimension mismatch issues. Reshaping data to match model...")
                    
                    # Calculate required n_times to match the model's expected sequence length
                    required_n_times = model_seq_len * 32
                    
                    if required_n_times > n_times:
                        # Need to pad data
                        padding = required_n_times - n_times
                        print(f"Padding data from {n_times} to {required_n_times} time points")
                        X_padded = np.zeros((n_trials, n_channels, required_n_times))
                        X_padded[:, :, :n_times] = X_raw
                        X_raw = X_padded
                    else:
                        # Need to truncate data
                        print(f"Truncating data from {n_times} to {required_n_times} time points")
                        X_raw = X_raw[:, :, :required_n_times]
                    
                    # Update n_times to reflect changed dimensions
                    _, _, n_times = X_raw.shape
        
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
        if custom_class_weights is not None:
            class_weights = custom_class_weights
            print(f"Using provided class weights: {class_weights}")
        else:
        class_counts = np.bincount(y)
        class_weights = torch.FloatTensor(
            np.sqrt(np.sum(class_counts) / (len(class_counts) * class_counts))
        ).to(self.device)
            print(f"Using default class weights: {class_weights}")
        
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
        
        # Variables to track if model is stuck predicting one class
        stuck_on_one_class = False
        stuck_counter = 0
        # New variables for adaptive balanced prediction
        apply_balancing = self.force_balanced_val  # Start with user setting
        balancing_strength = 0.0  # Will be increased if model gets stuck
        
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
                    
                    # Forward pass with balancing flag if enabled
                    outputs = self.model(X_raw_batch, X_csp_batch, X_freq_batch, force_balanced=apply_balancing)
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
                        
                        # Forward pass with balancing flag if enabled
                        outputs = self.model(X_raw_batch, X_csp_batch, X_freq_batch, force_balanced=apply_balancing)
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
                
                # Check if model is stuck predicting only one class
                # Get class distribution of predictions
                pred_counts = np.bincount(all_val_preds, minlength=self.n_classes)
                
                # Check if all or most predictions are for a single class
                if max(pred_counts) > 0.9 * len(all_val_preds):  # 90% or more are one class
                    majority_class = np.argmax(pred_counts)
                    print(f"  ! WARNING: Model is predicting class {majority_class} for {pred_counts[majority_class]}/{len(all_val_preds)} samples.")
                    
                    # If stuck on one class for multiple epochs, apply corrective actions
                    if stuck_on_one_class:
                        stuck_counter += 1
                        
                        # After 3 epochs stuck, try to nudge the model
                        if stuck_counter >= 3 and stuck_counter % 3 == 0:
                            # Apply more aggressive corrective action:
                            # 1. Increase learning rate temporarily
                            old_lr = optimizer.param_groups[0]['lr']
                            new_lr = old_lr * 2.0
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = new_lr
                            
                            print(f"  ! Taking corrective action: Increasing learning rate from {old_lr:.6f} to {new_lr:.6f}")
                            
                            # 2. Enable balanced prediction for validation
                            if not apply_balancing:
                                apply_balancing = True
                                print(f"  ! Enabling balanced prediction for validation")
                            
                            # 3. Add bias correction to the final layer
                            if hasattr(self.model, 'classifier') and isinstance(self.model.classifier, nn.Sequential):
                                # Find final linear layer
                                for module in self.model.classifier:
                                    if isinstance(module, nn.Linear) and module.out_features == self.n_classes:
                                        # Adjust the bias of the final layer to penalize the majority class
                                        with torch.no_grad():
                                            bias_adjustment = torch.zeros_like(module.bias)
                                            bias_adjustment[majority_class] = -0.5  # Penalize majority class
                                            other_classes = [c for c in range(self.n_classes) if c != majority_class]
                                            for c in other_classes:
                                                bias_adjustment[c] = 0.5 / len(other_classes)  # Boost other classes
                                            
                                            # Apply the adjustment
                                            module.bias.add_(bias_adjustment)
                                            print(f"  ! Added bias correction: {bias_adjustment}")
                    else:
                        # First time detecting this issue
                        stuck_on_one_class = True
                        stuck_counter = 1
                        # Enable balanced validation on first detection
                        apply_balancing = True
                        print(f"  ! Enabling balanced prediction for validation")
                else:
                    # Reset if the model recovered
                    if stuck_on_one_class:
                        print("  ✓ Model is no longer stuck predicting one class")
                    stuck_on_one_class = False
                    stuck_counter = 0
                
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
        
        # Get results both with and without balanced prediction
        results = {}
        for balanced_mode in [False, True]:
            mode_preds = []
            mode_probs = []
        
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
                
                    # Forward pass with balanced flag
                    outputs = self.model(X_raw_batch, X_csp_batch, X_freq_batch, force_balanced=balanced_mode)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                # Store results
                    mode_preds.extend(predicted.cpu().numpy())
                    mode_probs.extend(probabilities.cpu().numpy())
                    
                    # Only store targets once
                    if not balanced_mode:
                all_targets.extend(y_batch.cpu().numpy())
            
            # Store results for this mode
            if balanced_mode:
                results['balanced_preds'] = np.array(mode_preds)
                results['balanced_probs'] = np.array(mode_probs)
            else:
                all_preds = mode_preds
                all_probs = mode_probs
        
        # Calculate metrics for standard predictions
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        
        accuracy = accuracy_score(all_targets, all_preds)
        balanced_acc = balanced_accuracy_score(all_targets, all_preds)
        cm = confusion_matrix(all_targets, all_preds)
        
        # Calculate metrics for balanced predictions
        if 'balanced_preds' in results:
            balanced_preds = results['balanced_preds']
            balanced_accuracy = accuracy_score(all_targets, balanced_preds)
            balanced_balanced_acc = balanced_accuracy_score(all_targets, balanced_preds)
            balanced_cm = confusion_matrix(all_targets, balanced_preds)
        
        # Calculate per-class metrics
        class_accuracies = {}
        for i in range(self.n_classes):
            mask = all_targets == i
            if np.sum(mask) > 0:
                class_accuracies[i] = accuracy_score(all_targets[mask], all_preds[mask])
            else:
                class_accuracies[i] = float('nan')
        
        print(f"\nEvaluation results:")
        print(f"Standard prediction mode:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Balanced accuracy: {balanced_acc:.4f}")
        print(f"  Confusion matrix:\n{cm}")
        
        if 'balanced_preds' in results:
            print(f"\nBalanced prediction mode:")
            print(f"  Accuracy: {balanced_accuracy:.4f}")
            print(f"  Balanced accuracy: {balanced_balanced_acc:.4f}")
            print(f"  Confusion matrix:\n{balanced_cm}")
        
        for i in range(self.n_classes):
            print(f"Class {i} accuracy: {class_accuracies.get(i, float('nan')):.4f}")
        
        # Decide which predictions to return based on performance
        final_preds = all_preds
        final_balanced_acc = balanced_acc
        if 'balanced_preds' in results and balanced_balanced_acc > balanced_acc:
            print(f"\nNOTE: Using balanced prediction mode results (better balanced accuracy)")
            final_preds = results['balanced_preds']
            final_balanced_acc = balanced_balanced_acc
        
        return {
            'accuracy': accuracy,
            'balanced_accuracy': final_balanced_acc,
            'confusion_matrix': cm,
            'predictions': final_preds,
            'targets': all_targets,
            'probabilities': all_probs,
            'class_accuracies': class_accuracies,
            # Include balanced results if available
            'balanced_results': results if 'balanced_preds' in results else None
        }


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
                             n_epochs=100, save_models=True, val_split=0.4):
    """
    Load a pretrained base model and fine-tune it for a specific subject.
    Tests different augmentation strategies for fine-tuning.
    
    Args:
        base_model_path: Path to the pretrained model
        subject_id: Subject ID to fine-tune on
        run_ids: Run IDs to use
        n_epochs: Number of epochs for fine-tuning
        save_models: Whether to save the fine-tuned models
        val_split: Percentage of data to use for validation (default: 0.4 for fine-tuning)
        
    Returns:
        Dictionary of results for different augmentation strategies
    """
    print(f"Fine-tuning for subject {subject_id}...")
    print(f"Base model path: {base_model_path}")
    print(f"Testing {len(RUN_IDS)} run IDs: {RUN_IDS}")
    
    # Extract model configuration for fallback
    model_config = None
    metadata = {}
    
    # Try to extract model config without loading the full model
    try:
        print("Extracting model configuration...")
        save_dict = torch.load(base_model_path, map_location='cpu')
        model_config = save_dict.get('model_config', save_dict.get('config', {}))
        metadata = save_dict.get('metadata', {})
        print("Successfully extracted model configuration from checkpoint")
    except Exception as e:
        print(f"Could not extract model configuration: {str(e)}")
        print("Will continue with base model loading")
    
    # Load subject data
    try:
        print(f"Loading data for subject {subject_id}...")
    X, y, info = load_subject_data(subject_id=subject_id, run_ids=run_ids)
        print(f"Successfully loaded data for subject {subject_id}")
    except Exception as e:
        print(f"Error loading subject data: {str(e)}")
        raise ValueError(f"Could not load data for subject {subject_id}")
    
    # Print dataset information
    print(f"Data shape: {X.shape} (trials, channels, time points)")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Evaluate class balance
    class_counts = np.bincount(y)
    min_class = min(class_counts)
    total_samples = len(y)
    val_samples = int(total_samples * val_split)
    
    print(f"Using validation split of {val_split:.0%} ({val_samples} samples)")
    print(f"Smallest class has {min_class} samples")
    
    if val_samples < 10:
        print(f"WARNING: Very small validation set ({val_samples} samples). Results may have high variance.")
    
    # Define augmentation strategies to test for fine-tuning
    aug_strategies = [
        {
            'name': 'None', 
            'augmenter': None,  # Use None instead of NoAugmenter
            'apply_augmentation': False,  # Explicitly disable augmentation
            'force_balanced_val': True  # Enable balanced validation for all strategies
        },
        {
            'name': 'Optimized',
            'augmenter': None,  # Will be created for each run
            'apply_augmentation': True,
            'force_balanced_val': True
        },
        {
            'name': 'Basic',
            'augmenter': None,  # Will be created for each run
            'apply_augmentation': True,
            'methods': ['add_noise', 'scale_amplitude'],
            'force_balanced_val': True
        }
    ]
    
    results = {
        'strategies': [s['name'] for s in aug_strategies],
        'accuracy': [],
        'balanced_accuracy': [],
        'histories': [],
        'confusion_matrices': []
    }
    
    # Track which strategies have been completed
    completed_strategies = []
    
    for i, strategy in enumerate(aug_strategies):
        print(f"\n{'='*50}")
        print(f"STARTING STRATEGY {i+1}/{len(aug_strategies)}: {strategy['name']} augmentation")
        print(f"{'='*50}")
        
        # Clear GPU memory between strategies
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Cleared GPU cache")
        
        try:
            # Load the base model specifically for this strategy
            print(f"Loading base model for '{strategy['name']}' strategy...")
            try:
                base_classifier, strategy_metadata = HybridModelClassifier.load_model(base_model_path)
                # Enable balanced validation
                base_classifier.force_balanced_val = strategy.get('force_balanced_val', True)
                print(f"Successfully loaded base model for '{strategy['name']}' strategy")
                print(f"Balanced validation {'enabled' if base_classifier.force_balanced_val else 'disabled'}")
                
                # Check if the subject was in the training set
                if 'training_subjects' in strategy_metadata and subject_id in strategy_metadata['training_subjects']:
                    print(f"WARNING: Subject {subject_id} was used in training the base model!")
            except Exception as e:
                print(f"Error loading base model for '{strategy['name']}' strategy: {str(e)}")
                print(f"Will attempt to create a new model from configuration")
                
                if model_config is not None:
                    # Use the extracted configuration if available
                    n_classes = len(np.unique(y))
                    base_classifier = HybridModelClassifier(
                        n_classes=n_classes,
                        embedding_dim=model_config.get('embedding_dim', 64),
                        n_heads=model_config.get('n_heads', 4),
                        n_layers=model_config.get('n_layers', 2),
                        dropout=model_config.get('dropout', 0.3),
                        lr=0.0005,
                        batch_size=16,
                        n_epochs=n_epochs,
                        weight_decay=0.01,
                        augmentation_factor=1,  # Use minimal augmentation for stability
                        use_csp=model_config.get('use_csp', True),
                        use_freq=model_config.get('use_freq', True),
                        n_csp_components=model_config.get('n_csp_components', 4),
                        force_balanced_val=strategy.get('force_balanced_val', True)
                    )
                    print(f"Created new classifier from configuration")
                    print(f"Balanced validation {'enabled' if base_classifier.force_balanced_val else 'disabled'}")
                else:
                    print(f"Cannot continue with '{strategy['name']}' strategy - no model configuration available")
                    raise ValueError("No model configuration available")
        
        # Set up the augmenter based on strategy
        if strategy['name'] == 'Optimized':
            # Create optimized augmenter
            class OptimizedAugmenter(EEGDataAugmenter):
                def __init__(self, noise_level=0.03, scale_range=(0.9, 1.1), chunk_max_offset=30):
                    super().__init__(noise_level, scale_range, chunk_max_offset, support_variable_length=False)
                        print("Initializing OptimizedAugmenter")
                    
                        # Only use the simplest, most reliable augmentation methods to avoid errors
                        try:
                    self.beneficial_methods = {
                        'add_noise': self.add_noise,
                                'scale_amplitude': self.scale_amplitude
                            }
                            print(f"Successfully registered {len(self.beneficial_methods)} augmentation methods")
                        except Exception as e:
                            print(f"Error setting up augmentation methods: {str(e)}")
                            # Provide fallback empty dictionary
                            self.beneficial_methods = {}
                        
                        # Initialize trial cache as empty
                        self.trial_cache = []
                        self.trial_labels = []
                
                def augment(self, x, y=None):
                        """Apply simplified augmentation strategy"""
                        try:
                            # Start with a copy of the input
                    x_aug = x.copy()
                    
                            # Apply simple noise augmentation
                            noise = np.random.normal(0, self.noise_level, x_aug.shape)
                            x_aug = x_aug + noise
                            
                            # Apply simple amplitude scaling
                            scale = np.random.uniform(self.scale_min, self.scale_max)
                            x_aug = x_aug * scale
                            
                            return x_aug
                        except Exception as e:
                            print(f"Error in augment: {str(e)}")
                            # Return the original data if anything fails
                            return x
                        
                    def augment_batch(self, X, y, augmentation_factor=1):
                        """Generate augmented data for a batch of EEG trials - simplified version"""
                        try:
                            print(f"OptimizedAugmenter.augment_batch called with X shape {X.shape}, y shape {y.shape}, factor {augmentation_factor}")
                            
                            # Always start with the original data
                            n_trials = X.shape[0]
                            augmented_X = [X]
                            augmented_y = [y]
                            
                            # Only add 1 augmentation to avoid potential issues
                            # Generate augmented data one trial at a time
                            augmented_trials = []
                            for i in range(n_trials):
                                try:
                                    # Apply simple noise augmentation to this trial
                                    trial = X[i].copy()
                                    noise = np.random.normal(0, self.noise_level, trial.shape)
                                    trial_aug = trial + noise
                                    augmented_trials.append(trial_aug)
                                except Exception as e:
                                    print(f"Error augmenting trial {i}: {str(e)}")
                                    # Use original trial as fallback
                                    augmented_trials.append(X[i].copy())
                            
                            # Stack the augmented trials into a batch
                            X_batch_aug = np.stack(augmented_trials, axis=0)
                            
                            # Add to our collection
                            augmented_X.append(X_batch_aug)
                            augmented_y.append(y)
                            
                            # Try to concatenate, with fallback to original data
                            try:
                                X_combined = np.concatenate(augmented_X, axis=0)
                                y_combined = np.concatenate(augmented_y, axis=0)
                                print(f"Successfully created augmented batch: X shape {X_combined.shape}, y shape {y_combined.shape}")
                                return X_combined, y_combined
                            except Exception as e:
                                print(f"Error concatenating augmented data: {str(e)}")
                                # Return original data as fallback
                                return X, y
                            
                        except Exception as e:
                            print(f"Error in augment_batch: {str(e)}")
                            import traceback
                            traceback.print_exc()
                            # Return the original data if anything fails
                            return X, y
            
            strategy['augmenter'] = OptimizedAugmenter()
                print(f"Created {strategy['name']} augmenter: {strategy['augmenter']}")
            
        elif strategy['name'] == 'Basic':
            # Create basic augmenter with limited methods
            class BasicAugmenter(EEGDataAugmenter):
                def __init__(self, noise_level=0.03, scale_range=(0.9, 1.1), chunk_max_offset=30):
                    super().__init__(noise_level, scale_range, chunk_max_offset, support_variable_length=False)
                        print("Initializing BasicAugmenter")
                    
                        # Just the simplest methods for maximum reliability
                        try:
                    self.basic_methods = {
                                'add_noise': self.add_noise
                            }
                            print(f"Successfully registered {len(self.basic_methods)} augmentation methods")
                        except Exception as e:
                            print(f"Error setting up basic augmentation methods: {str(e)}")
                            # Provide fallback empty dictionary
                            self.basic_methods = {}
                        
                        # Initialize trial cache as empty
                        self.trial_cache = []
                        self.trial_labels = []
                
                def augment(self, x, y=None):
                        """Apply extremely basic augmentation strategy - just noise"""
                        try:
                            # Start with a copy of the input
                    x_aug = x.copy()
                    
                            # Apply simple noise augmentation
                            noise = np.random.normal(0, self.noise_level, x_aug.shape)
                            x_aug = x_aug + noise
                            
                            return x_aug
                        except Exception as e:
                            print(f"Error in augment: {str(e)}")
                            # Return the original data if anything fails
                            return x
                        
                    def augment_batch(self, X, y, augmentation_factor=1):
                        """Generate augmented data for a batch of EEG trials - simplified version"""
                        try:
                            print(f"BasicAugmenter.augment_batch called with X shape {X.shape}, y shape {y.shape}")
                            
                            # Always start with the original data
                            n_trials = X.shape[0]
                            
                            # Only add one copy of noise-augmented data
                            # Apply simple noise to the entire batch at once
                            X_aug = X.copy()
                            noise = np.random.normal(0, self.noise_level, X_aug.shape)
                            X_aug = X_aug + noise
                            
                            # Combine original and augmented data
                            X_combined = np.concatenate([X, X_aug], axis=0)
                            y_combined = np.concatenate([y, y], axis=0)
                            
                            print(f"Successfully created basic augmented batch: X shape {X_combined.shape}, y shape {y_combined.shape}")
                            return X_combined, y_combined
                            
                        except Exception as e:
                            print(f"Error in basic augment_batch: {str(e)}")
                            import traceback
                            traceback.print_exc()
                            # Return the original data if anything fails
                            return X, y
            
            strategy['augmenter'] = BasicAugmenter()
                print(f"Created {strategy['name']} augmenter: {strategy['augmenter']}")
        
        # Set the augmenter
            if strategy['name'] != 'None':
        base_classifier.augmenter = strategy['augmenter']
                print(f"Set augmenter for strategy '{strategy['name']}'")
            else:
                base_classifier.augmenter = None
                print(f"Using no augmentation for strategy '{strategy['name']}'")
        
        # Update classifier parameters for fine-tuning
        base_classifier.n_epochs = n_epochs
            base_classifier.lr = 0.0002  # Slightly higher initial learning rate for fine-tuning
        base_classifier.batch_size = 16  # Smaller batch size for subject-specific tuning
            base_classifier.weight_decay = 0.03  # Increased L2 regularization to combat overfitting
            
            # Fine-tune on the subject's data with increased validation split
            print(f"Fine-tuning model with {val_split:.0%} validation split...")
            
            # Set up custom training parameters only for this run
            optimizer_params = {
                'lr': base_classifier.lr,
                'weight_decay': base_classifier.weight_decay,
            }
            
            # Create a custom train_and_evaluate function that uses a more aggressive scheduler
            def custom_train_evaluate_with_better_scheduling(X, y, val_split):
                # Calculate class weights for more aggressive balancing
                # Check if y is a tensor or numpy array
                if isinstance(y, torch.Tensor):
                    y_numpy = y.cpu().numpy()
                else:
                    y_numpy = y
                    
                class_counts = np.bincount(y_numpy)
                # More aggressive weighting to help avoid the "predict all one class" issue
                weight_factor = 3.0  # Use an even higher factor (3.0 instead of 2.0)
                
                # Use a smoother power-based weighting formula
                total_samples = np.sum(class_counts)
                n_classes = len(class_counts)
                
                # Calculate inverse frequency with smoothing
                inv_freq = total_samples / (n_classes * class_counts + 1e-5)
                
                # Apply power function to make weights more aggressive
                class_weights = torch.FloatTensor(
                    np.power(inv_freq, weight_factor)
                ).to(base_classifier.device)
                
                # Normalize weights to avoid scaling issues
                class_weights = class_weights / class_weights.sum() * n_classes
                
                print(f"Using aggressive class weights: {class_weights}")

                # Reduce weight decay to avoid over-regularization
                # Use lower weight decay for fine-tuning
                base_classifier.weight_decay = 0.005  # Reduced from 0.01 to allow more flexibility
                
                # Increase initial learning rate slightly to help escape local minima
                original_lr = base_classifier.lr
                base_classifier.lr = max(original_lr * 1.5, 0.0003)  # At least 0.0003
                
                # Set a longer patience for early stopping to give model time to adapt
                patience = min(35, n_epochs // 3)  # Roughly 1/3 of total epochs
                
                print(f"Using weight_decay={base_classifier.weight_decay}, lr={base_classifier.lr}, patience={patience}")
                
                # Call the original method but with custom parameters
        train_results = base_classifier.train_and_evaluate(
                    X, y, 
                    val_split=val_split, 
                    early_stopping_patience=patience,
                    custom_class_weights=class_weights  # Pass custom class weights
                )
                return train_results
            
            # Train with custom parameters
            print(f"Starting training for strategy '{strategy['name']}'...")
            train_results = custom_train_evaluate_with_better_scheduling(X, y, val_split)
            print(f"Training completed for strategy '{strategy['name']}'")
        
        # Store results
        results['accuracy'].append(train_results['accuracy'])
        results['balanced_accuracy'].append(train_results['balanced_accuracy'])
        results['histories'].append(train_results['history'])
        results['confusion_matrices'].append(train_results['confusion_matrix'])
            
            # Track successful completion
            completed_strategies.append(strategy['name'])
            print(f"Successfully completed strategy {i+1}/{len(aug_strategies)}: '{strategy['name']}'")
        
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
    
        except Exception as e:
            print(f"Error during fine-tuning with {strategy['name']} strategy: {str(e)}")
            import traceback
            traceback.print_exc()  # Print full traceback for debugging
            print(f"Skipping {strategy['name']} strategy for this subject")
            
            # Add placeholder results
            results['accuracy'].append(float('nan'))
            results['balanced_accuracy'].append(float('nan'))
            results['histories'].append({'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_balanced_acc': []})
            results['confusion_matrices'].append(np.zeros((2, 2)))
            
            # Force Python to collect garbage
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Only compare strategies if we have at least one successful run
    print(f"\n{'='*50}")
    print(f"SUMMARIZING RESULTS FOR ALL STRATEGIES")
    print(f"{'='*50}")
    print(f"Completed strategies: {completed_strategies}")
    
    if len(completed_strategies) > 0:
    # Compare strategies
        try:
    plt.figure(figsize=(10, 6))
    x = np.arange(len(results['strategies']))
    width = 0.35
    
            # Filter out NaN values for plotting
            accuracies = np.array(results['accuracy'])
            balanced_accuracies = np.array(results['balanced_accuracy'])
            
            plt.bar(x - width/2, accuracies, width, 
            label='Accuracy', color='royalblue', alpha=0.7)
            plt.bar(x + width/2, balanced_accuracies, width, 
            label='Balanced Accuracy', color='darkorange', alpha=0.7)
    
    plt.xlabel('Augmentation Strategy')
    plt.ylabel('Validation Accuracy')
    plt.title(f'Subject {subject_id} - Fine-tuning Results')
    plt.xticks(x, results['strategies'])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
        except Exception as e:
            print(f"Error plotting summary: {str(e)}")
    
    # Print summary
    print("\nFine-tuning Results Summary:")
    print(f"{'Strategy':<15} {'Accuracy':<15} {'Balanced Accuracy':<15}")
    print('-' * 45)
    
    for i, strategy in enumerate(results['strategies']):
            try:
                acc = results['accuracy'][i]
                bal_acc = results['balanced_accuracy'][i]
                acc_str = f"{acc:.4f}" if not np.isnan(acc) else "FAILED"
                bal_acc_str = f"{bal_acc:.4f}" if not np.isnan(bal_acc) else "FAILED"
                print(f"{strategy:<15} {acc_str:<15} {bal_acc_str:<15}")
            except Exception as e:
                print(f"{strategy:<15} ERROR          ERROR")
        
        # Find the best strategy among successful ones
        try:
            valid_indices = [i for i, acc in enumerate(results['balanced_accuracy']) 
                             if i < len(results['balanced_accuracy']) and not np.isnan(acc)]
            
            if valid_indices:
                best_idx = valid_indices[np.argmax([results['balanced_accuracy'][i] for i in valid_indices])]
    best_strategy = results['strategies'][best_idx]
    print(f"\nBest augmentation strategy for Subject {subject_id}: {best_strategy} "
          f"(Balanced Accuracy: {results['balanced_accuracy'][best_idx]:.4f})")
            else:
                print(f"\nNo successful augmentation strategy for Subject {subject_id}")
        except Exception as e:
            print(f"Error finding best strategy: {str(e)}")
    else:
        print(f"All fine-tuning strategies failed for Subject {subject_id}")
    
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
        base_model_path = os.path.join(model_dir, f'{base_model_name}_{metadata["requested_n_subjects"]}_subjects.pt')
    else:
        print(f"Using existing base model: {base_model_path}")
    
    # # Step 2: Fine-tune for individual test subjects
    # fine_tune_results = {}
    # for subject in test_subjects[0:1]:
    #     try:
    #         # Fine-tune for this test subject
    #         print(f"\nFine-tuning for subject {subject}...")
    #         subject_results = finetune_subject_specific(
    #             base_model_path=base_model_path,
    #             subject_id=subject,
    #             run_ids=RUN_IDS,
    #             n_epochs=100,
    #             save_models=True
    #         )
    #         fine_tune_results[subject] = subject_results
    #     except Exception as e:
    #         print(f"Error fine-tuning subject {subject}: {str(e)}")
    #         print("Continuing with next subject...")
    #         continue
    
    plt.show()
