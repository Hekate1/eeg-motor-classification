# src/feature_modules.py
from feature_utils import CSPTransformer
from hycnn_blocks import SpecCNN
import torch
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaseFeatureModule:
    """
    Base interface for feature modules.
    """
    def fit(self, X, y, subject_indices=None):
        """Fit the feature module on training data."""
        raise NotImplementedError

    def transform(self, X, subject_indices=None):
        """Transform data to feature representation."""
        raise NotImplementedError

    @property
    def feature_dim(self):
        """Return the dimension of the output feature vector."""
        raise NotImplementedError

class CSPModule(BaseFeatureModule):
    """
    CSP feature extraction module wrapping the existing CSPTransformer.
    """
    def __init__(self, n_components=4, per_subject=True, filter_bank=False, bands=None, sfreq=160):
        self.n_components = n_components
        self.per_subject = per_subject
        self.filter_bank = filter_bank
        self.bands = bands
        self.sfreq = sfreq
        self.transformer = None

    def fit(self, X, y, subject_indices=None):
        # Instantiate and fit the underlying CSPTransformer
        X = X.detach().cpu().numpy().astype(np.float64)
        y = y.detach().cpu().numpy()
        if subject_indices is not None:
            subject_indices = subject_indices.detach().cpu().numpy()
        self.transformer = CSPTransformer(
            X, y,
            per_subject=self.per_subject,
            subject_indices=subject_indices,
            n_components=self.n_components,
            filter_bank=self.filter_bank,
            bands=self.bands,
            sfreq=self.sfreq
        )

    def transform(self, X, subject_indices=None):
        if self.transformer is None:
            raise RuntimeError("CSPModule must be fit before calling transform.")
        # Handle torch.Tensor input: convert to NumPy, apply transform, then convert back
        tensor_input = torch.is_tensor(X)
        if tensor_input:
            X_np = X.detach().cpu().numpy().astype(np.float64)
            if subject_indices is not None and torch.is_tensor(subject_indices):
                subj_idx = subject_indices.detach().cpu().numpy()
            else:
                subj_idx = subject_indices
        else:
            X_np = X
            subj_idx = subject_indices
        out = self.transformer.transform(X_np, subj_idx)
        if tensor_input:
            # Return CPU tensor to avoid GPU round-trips; training loop will move all features once
            return torch.from_numpy(out)
        return out

    @property
    def feature_dim(self):
        # Determine feature dimension based on configuration
        if self.bands and self.filter_bank:
            return len(self.bands) * self.n_components
        return self.n_components

class SpectrogramModule(BaseFeatureModule):
    """
    Spectrogram feature extraction module using log-power STFTs and SpecCNN.
    """
    def __init__(self, n_channels, n_fft=64, hop_length=32, hidden=64, out_dim=128, dropout=0.2):
        # Number of EEG channels
        self.n_channels = n_channels
        # STFT parameters
        self.n_fft = n_fft
        self.hop_length = hop_length or (n_fft // 2)
        self.dropout = dropout
        self.hidden = hidden
        # Initialize SpecCNN with the correct number of channels
        self.spec_cnn = SpecCNN(in_ch=self.n_channels, hidden=hidden, out_dim=out_dim, dropout=dropout)
        # Ensure the SpecCNN weights are on the correct device
        self.spec_cnn.to(DEVICE)
        self.out_dim = out_dim

    def fit(self, X, y, subject_indices=None):
        # No fitting required for spectrogram features
        pass

    def transform(self, X, subject_indices=None):
        """
        Compute log-power STFT and extract features via SpecCNN.
        X: numpy array of shape [n_samples, n_channels, n_times]
        Returns: numpy array shape [n_samples, out_dim]
        """
        # Convert to torch tensor (handle numpy arrays or Tensors)
        if torch.is_tensor(X):
            X_tensor = X.float().to(DEVICE)
        else:
            X_tensor = torch.from_numpy(X).float().to(DEVICE)
        B, C, T = X_tensor.shape
        # Compute STFT and extract features for full batch
        X_flat = X_tensor.reshape(B * C, T)
        spec = torch.stft(X_flat, self.n_fft, self.hop_length, pad_mode='constant', return_complex=True)
        power = spec.abs() ** 2
        log_power = torch.log1p(power)
        freq_bins, time_frames = log_power.size(1), log_power.size(2)
        log_power = log_power.view(B, C, freq_bins, time_frames)
        with torch.no_grad():
            feats = self.spec_cnn(log_power)
        # Release intermediate tensors
        del X_flat, spec, power, log_power
        return feats

    @property
    def feature_dim(self):
        return self.out_dim 