import numpy as np
import torch

DEBUG = False

# Debug print function
def dprint(*args, **kwargs):
    """Only print if DEBUG is True"""
    if DEBUG:
        print(*args, **kwargs)

class EEGDataAugmenter:
    """
    Class for applying data augmentation techniques to EEG data
    (Modified from single_subject_lstm.py)
    """
    def __init__(self, noise_level=0.03, scale_range=(0.9, 1.1),
                 chunk_max_offset=30, support_variable_length=True,
                 augmentation_factor=3):
        """
        Initialize the data augmenter with various augmentation parameters
        
        Args:
            noise_level: Standard deviation of Gaussian noise to add
            scale_range: Range for random amplitude scaling
            chunk_max_offset: Maximum offset for chunk selection
            support_variable_length: Whether to support variable-length outputs
            augmentation_factor: Number of augmented samples to generate per original sample
        """
        self.noise_level = noise_level
        self.scale_range = scale_range
        self.chunk_max_offset = chunk_max_offset
        self.support_variable_length = support_variable_length
        self.augmentation_factor = augmentation_factor
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
        # Check if trial cache exists and has the right structure
        assert self.trial_cache is not None and isinstance(self.trial_cache, dict) and 'data' in self.trial_cache
        
        # For class-matching, check if we have labels
        if same_class_only:
            assert y is not None and self.trial_cache.get('labels') is not None
            
            # Find trials of the same class
            same_class_indices = np.where(self.trial_cache['labels'] == y)[0]
            assert len(same_class_indices) > 0
            mix_idx = np.random.choice(same_class_indices)
        else:
            # Select any random trial
            mix_idx = np.random.randint(0, len(self.trial_cache['data']))
        
        # Get the trial to mix with - now correctly accessing the 'data' field
        x2 = self.trial_cache['data'][mix_idx]
        
        assert x.shape == x2.shape
        
        # Generate mixing ratio from Beta distribution
        mix_ratio = np.random.beta(alpha, alpha)
        
        # Mix the trials
        return mix_ratio * x + (1 - mix_ratio) * x2
    
    def augment(self, x, y=None):
        """Apply multiple augmentation techniques to a single trial"""
        # Available augmentations
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

    def augment_batch(self, X, y):
        """
        Base method to be implemented by subclasses.
        Creates augmented versions of the input data.
        
        Args:
            X: Input data with shape (n_trials, n_channels, n_timesteps)
            y: Labels with shape (n_trials,)
                                 
        Returns:
            X_aug: Augmented data
            y_aug: Corresponding labels
        """
        raise NotImplementedError("Subclasses must implement this method")

class NoAugmenter(EEGDataAugmenter):
    def augment(self, x, y=None):
        return x
    
    def augment_batch(self, X, y):
        return X, y

class OptimizedAugmenter(EEGDataAugmenter):
    """
    Optimized augmentation strategy using only the most beneficial methods
    for EEG motor imagery data
    """
    def __init__(self, augmentation_factor=5, noise_level=0.03, scale_range=(0.9, 1.1), chunk_max_offset=30):
        super().__init__(noise_level, scale_range, chunk_max_offset, support_variable_length=False, augmentation_factor=augmentation_factor)
        
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
    
    def augment_batch(self, X, y):
        """
        Augment a batch of data
        
        Args:
            X: Input data [n_samples, n_channels, n_times]
            y: Labels [n_samples]
            
        Returns:
            X_aug: Augmented data [n_samples * (augmentation_factor + 1), n_channels, n_times]
            y_aug: Augmented labels [n_samples * (augmentation_factor + 1)]
        """
        # Handle torch.Tensor inputs by converting to numpy and remembering to convert back
        tensor_input = torch.is_tensor(X)
        if tensor_input:
            device = X.device
            X = X.detach().cpu().numpy()
            y = y.detach().cpu().numpy() if y is not None else None
        n_samples = X.shape[0]
        
        # Store original samples for mixup
        self.trial_cache = {
            'data': X.copy(),
            'labels': y.copy() if y is not None else None
        }
        
        # Initialize augmented data arrays
        X_aug = np.zeros((n_samples * (self.augmentation_factor + 1),) + X.shape[1:], dtype=X.dtype)
        y_aug = np.zeros(n_samples * (self.augmentation_factor + 1), dtype=int if y is not None else float)
        
        # Copy original samples
        X_aug[:n_samples] = X
        if y is not None:
            y_aug[:n_samples] = y
        
        # Generate augmented samples
        for i in range(n_samples):
            for j in range(self.augmentation_factor):
                aug_idx = n_samples + i * self.augmentation_factor + j
                X_aug[aug_idx] = self.augment(X[i], y[i] if y is not None else None)
                if y is not None:
                    y_aug[aug_idx] = y[i]
        
        # Convert back to torch if needed
        if tensor_input:
            X_aug = torch.from_numpy(X_aug).to(device)
            y_aug = torch.from_numpy(y_aug).to(device)
        return X_aug, y_aug


class BasicAugmenter(EEGDataAugmenter):
    """
    Basic augmentation strategy using only simple methods
    """
    def __init__(self, augmentation_factor=5, noise_level=0.03, scale_range=(0.9, 1.1), chunk_max_offset=30):
        super().__init__(noise_level, scale_range, chunk_max_offset, support_variable_length=False, augmentation_factor=augmentation_factor)
        
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
    
    def augment_batch(self, X, y):
        """
        Augment a batch of data
        
        Args:
            X: Input data [n_samples, n_channels, n_times]
            y: Labels [n_samples]
            
        Returns:
            X_aug: Augmented data [n_samples * (augmentation_factor + 1), n_channels, n_times]
            y_aug: Augmented labels [n_samples * (augmentation_factor + 1)]
        """
        # Handle torch.Tensor inputs by converting to numpy and remembering to convert back
        tensor_input = torch.is_tensor(X)
        if tensor_input:
            device = X.device
            X = X.detach().cpu().numpy()
            y = y.detach().cpu().numpy() if y is not None else None
        n_samples = X.shape[0]
        
        # Store original samples for potential mixup
        self.trial_cache = {
            'data': X.copy(),
            'labels': y.copy() if y is not None else None
        }
        
        # Initialize augmented data arrays
        X_aug = np.zeros((n_samples * (self.augmentation_factor + 1),) + X.shape[1:], dtype=X.dtype)
        y_aug = np.zeros(n_samples * (self.augmentation_factor + 1), dtype=int if y is not None else float)
        
        # Copy original samples
        X_aug[:n_samples] = X
        if y is not None:
            y_aug[:n_samples] = y
        
        # Generate augmented samples
        for i in range(n_samples):
            for j in range(self.augmentation_factor):
                aug_idx = n_samples + i * self.augmentation_factor + j
                X_aug[aug_idx] = self.augment(X[i], y[i] if y is not None else None)
                if y is not None:
                    y_aug[aug_idx] = y[i]
        
        # Convert back to torch if needed
        if tensor_input:
            X_aug = torch.from_numpy(X_aug).to(device)
            y_aug = torch.from_numpy(y_aug).to(device)
        return X_aug, y_aug

class SingleSubjectAugmenter(EEGDataAugmenter):
    """
    Specialized augmenter for single-subject fine-tuning with stronger augmentations
    that preserve subject-specific characteristics while creating realistic variations.
    """
    def __init__(self, noise_level=0.05, scale_range=(0.85, 1.15), chunk_max_offset=40,
                 augmentation_strategies=None, strategy_weights=None, visualization_mode=False,
                 augmentation_factor=10):
        super().__init__(noise_level, scale_range, chunk_max_offset, 
                         support_variable_length=False, augmentation_factor=augmentation_factor)
        
        # Enhanced parameters for visualization if needed
        self.visualization_mode = visualization_mode
        
        # Define all available augmentation methods
        self.all_methods = {
            'add_noise': self.add_noise,
            'scale_amplitude': self.scale_amplitude,
            'spectral_perturbation': self.spectral_perturbation,
            'smooth_warping': self.smooth_warping,
            'frequency_band_noise': self.frequency_band_noise,
            'channel_dropout': self.channel_dropout,
            'temporal_shift': self.temporal_shift,
            'neural_jitter': self.neural_jitter,
            'trial_mixup': None  # Special handling
        }
        
        # Set active strategies (default: all)
        self.active_strategies = augmentation_strategies
        if self.active_strategies is None:
            self.active_strategies = list(self.all_methods.keys())
        
        # Set strategy weights for random selection
        self.strategy_weights = strategy_weights
        if self.strategy_weights is None:
            self.strategy_weights = [1.0] * len(self.active_strategies)
        
        # Normalize weights
        if sum(self.strategy_weights) > 0:
            self.strategy_weights = [w / sum(self.strategy_weights) for w in self.strategy_weights]
    
    def add_noise(self, x):
        """Add random Gaussian noise to the signal with adaptive scaling"""
        # Check if input signal is very small in magnitude
        data_range = np.max(x) - np.min(x)
        is_small_signal = data_range < 0.001
        
        # Adapt noise level based on signal magnitude
        if is_small_signal or self.visualization_mode:
            # For visualization or small signals, use stronger noise
            noise_level = self.noise_level * 5 if self.visualization_mode else self.noise_level * 2
        else:
            noise_level = self.noise_level
            
        # Scale noise by the standard deviation of the signal for better adaptation
        # For very small signals, establish a minimum noise level
        noise_scale = max(noise_level * np.std(x), 1e-5 if is_small_signal else 0)
        noise = np.random.normal(0, noise_scale, x.shape)
        return x + noise
    
    def scale_amplitude(self, x):
        """Scale the amplitude of the signal randomly"""
        if self.visualization_mode:
            # Use more extreme scaling for visualization
            scale_range = (0.7, 1.3)
        else:
            scale_range = self.scale_range
            
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return x * scale
    
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
        
        # Adjust noise intensity for visualization
        intensity_factor = 5 if self.visualization_mode else 1
        
        for c in range(n_channels):
            # Convert to frequency domain
            x_fft = np.fft.rfft(x[c])
            freqs = np.fft.rfftfreq(n_times, d=1/fs)
            
            # Target mu (8-12Hz) and beta (13-30Hz) bands for motor imagery
            mu_mask = (freqs >= 8) & (freqs <= 12)
            beta_mask = (freqs >= 13) & (freqs <= 30)
            
            # Add noise only to these frequency bands
            mu_noise = np.random.normal(0, 0.1 * intensity_factor, size=sum(mu_mask)) * np.abs(x_fft[mu_mask].mean())
            beta_noise = np.random.normal(0, 0.05 * intensity_factor, size=sum(beta_mask)) * np.abs(x_fft[beta_mask].mean())
            
            # Apply noise
            x_fft[mu_mask] = x_fft[mu_mask] * (1 + mu_noise)
            x_fft[beta_mask] = x_fft[beta_mask] * (1 + beta_noise)
            
            # Convert back to time domain
            x_new[c] = np.fft.irfft(x_fft, n=n_times)
        
        return x_new
    
    def channel_dropout(self, x, dropout_rate=None):
        """
        Randomly set some channels to zero for short time periods
        This simulates temporary electrode connection issues
        """
        # Use stronger dropout for visualization
        if dropout_rate is None:
            dropout_rate = 0.05 if not self.visualization_mode else 0.2
            
        n_channels, n_times = x.shape
        x_new = x.copy()
        
        # Determine number of dropout events - more for visualization
        n_events = np.random.randint(1, 4 if self.visualization_mode else 3)
        
        for _ in range(n_events):
            # Randomly choose a channel
            channel = np.random.randint(0, n_channels)
            
            # Randomly choose a time segment
            segment_length = np.random.randint(
                10, 
                min(100 if self.visualization_mode else 50, n_times // 5)
            )
            start_time = np.random.randint(0, n_times - segment_length)
            
            # Apply dropout with tapering at edges for smoothness
            taper_length = min(10, segment_length // 4)
            
            # Create tapering windows
            if taper_length > 0:
                taper_in = np.linspace(1, dropout_rate, taper_length)
                taper_out = np.linspace(dropout_rate, 1, taper_length)
                
                # Apply tapering
                x_new[channel, start_time:start_time+taper_length] *= taper_in
                x_new[channel, start_time+segment_length-taper_length:start_time+segment_length] *= taper_out
                
                # Apply full dropout to middle section
                if segment_length > 2*taper_length:
                    x_new[channel, start_time+taper_length:start_time+segment_length-taper_length] *= dropout_rate
            else:
                # Just apply dropout to the whole segment
                x_new[channel, start_time:start_time+segment_length] *= dropout_rate
                
        return x_new
    
    def temporal_shift(self, x):
        """
        Apply different small time shifts to different channels
        This simulates variations in neural conduction and measurement delays
        """
        n_channels, n_times = x.shape
        x_new = np.zeros_like(x)
        
        # Apply a different small shift to each channel
        # Use larger shifts for visualization
        max_shift = 10 if self.visualization_mode else 5
        
        for c in range(n_channels):
            # Random shift between -max_shift and max_shift samples
            shift = np.random.randint(-max_shift, max_shift + 1)
            
            if shift == 0:
                x_new[c] = x[c]
            elif shift > 0:
                x_new[c, shift:] = x[c, :-shift]
            else:
                x_new[c, :shift] = x[c, -shift:]
                
        return x_new
    
    def neural_jitter(self, x):
        """
        Simulate natural variability in neural response timing
        This creates small temporal jitter in signal peaks/troughs
        """
        n_channels, n_times = x.shape
        x_new = x.copy()
        
        # Apply to each channel independently
        for c in range(n_channels):
            # Find local maxima and minima (peaks and troughs)
            peaks = np.where((np.diff(np.sign(np.diff(x[c]))) < 0))[0] + 1
            troughs = np.where((np.diff(np.sign(np.diff(x[c]))) > 0))[0] + 1
            
            # Filter to avoid edge points
            peaks = peaks[(peaks > 5) & (peaks < n_times - 5)]
            troughs = troughs[(troughs > 5) & (troughs < n_times - 5)]
            
            # Randomly jitter a subset of peaks and troughs
            for points in [peaks, troughs]:
                if len(points) == 0:
                    continue
                    
                # Select a random subset (30-50%) of points to jitter
                n_points = max(1, int(np.random.uniform(0.3, 0.5) * len(points)))
                selected_points = np.random.choice(points, n_points, replace=False)
                
                for point in selected_points:
                    # Small jitter (-2 to +2 samples)
                    jitter = np.random.randint(-2, 3)
                    
                    if jitter != 0:
                        # Simple approach: swap values
                        target_point = point + jitter
                        if 0 <= target_point < n_times:
                            x_new[c, point], x_new[c, target_point] = x_new[c, target_point], x_new[c, point]
        
        return x_new
    
    def augment(self, x, y=None):
        """
        Apply stronger, more varied augmentation for single-subject fine-tuning
        
        Args:
            x: Input data [n_channels, n_times]
            y: Label (optional)
            
        Returns:
            Augmented data
        """
        x_aug = x.copy()
        
        # Apply 2-4 random augmentations (more aggressive than before)
        num_augmentations = np.random.randint(2, 5)
        
        # Select strategies based on weights
        selected_strategies = np.random.choice(
            self.active_strategies, 
            size=min(num_augmentations, len(self.active_strategies)),
            replace=False,
            p=self.strategy_weights if self.strategy_weights else None
        )
        
        # Apply selected augmentations
        for strategy in selected_strategies:
            if strategy == 'trial_mixup':
                if self.trial_cache is not None and np.random.rand() < 0.7:  # Higher probability
                    if y is not None:
                        x_aug = self.trial_mixup(x_aug, y, same_class_only=True, alpha=0.3)
                    else:
                        # If no label provided, use any random trial (e.g. during visualization)
                        x_aug = self.trial_mixup(x_aug, y=0, same_class_only=False, alpha=0.3)
            elif strategy in self.all_methods and self.all_methods[strategy] is not None:
                x_aug = self.all_methods[strategy](x_aug)
            
        return x_aug
    
    def augment_batch(self, X, y):
        """
        Augment a batch of data with higher augmentation factor for single-subject fine-tuning
        
        Args:
            X: Input data [n_samples, n_channels, n_times]
            y: Labels [n_samples]
            
        Returns:
            X_aug: Augmented data [n_samples * (augmentation_factor + 1), n_channels, n_times]
            y_aug: Augmented labels [n_samples * (augmentation_factor + 1)]
        """
        # Handle torch.Tensor inputs by converting to numpy and remembering to convert back
        tensor_input = torch.is_tensor(X)
        if tensor_input:
            device = X.device
            X = X.detach().cpu().numpy()
            y = y.detach().cpu().numpy() if y is not None else None
        n_samples = X.shape[0]
        
        # Store original samples for mixup
        self.trial_cache = {
            'data': X.copy(),
            'labels': y.copy() if y is not None else None
        }
        
        # Initialize augmented data arrays
        X_aug = np.zeros((n_samples * (self.augmentation_factor + 1),) + X.shape[1:], dtype=X.dtype)
        y_aug = np.zeros(n_samples * (self.augmentation_factor + 1), dtype=int if y is not None else float)
        
        # Copy original samples
        X_aug[:n_samples] = X
        if y is not None:
            y_aug[:n_samples] = y
        
        # Generate augmented samples
        for i in range(n_samples):
            for j in range(self.augmentation_factor):
                aug_idx = n_samples + i * self.augmentation_factor + j
                X_aug[aug_idx] = self.augment(X[i], y[i] if y is not None else None)
                if y is not None:
                    y_aug[aug_idx] = y[i]
        
        # Convert back to torch if needed
        if tensor_input:
            X_aug = torch.from_numpy(X_aug).to(device)
            y_aug = torch.from_numpy(y_aug).to(device)
        return X_aug, y_aug
    
    def create_strategy_specific_augmenter(self, strategy_name):
        """
        Create an augmenter that only uses a single specified strategy
        
        Args:
            strategy_name: Name of the strategy to use
            
        Returns:
            A new SingleSubjectAugmenter that only applies the specified strategy
        """
        if strategy_name not in self.all_methods:
            raise ValueError(f"Strategy '{strategy_name}' not found in available methods")
        
        # Log the augmentation factor for debugging
        dprint(f"Creating strategy-specific augmenter for {strategy_name} with augmentation_factor={self.augmentation_factor}")
        
        # Create a new augmenter with only the specified strategy active
        return SingleSubjectAugmenter(
            noise_level=self.noise_level,
            scale_range=self.scale_range,
            chunk_max_offset=self.chunk_max_offset,
            augmentation_strategies=[strategy_name],
            strategy_weights=[1.0],
            visualization_mode=self.visualization_mode,
            augmentation_factor=self.augmentation_factor
        )

    def apply_single_augmentation(self, x, y=None, strategy_name=None):
        """
        Apply a single specified augmentation strategy
        
        Args:
            x: Input data [n_channels, n_times]
            y: Label (optional)
            strategy_name: Name of the strategy to apply
            
        Returns:
            Augmented data
        """
        if strategy_name is None or strategy_name not in self.all_methods:
            # If no strategy specified or invalid, choose randomly from active strategies
            strategy_name = np.random.choice(self.active_strategies, 
                                            p=self.strategy_weights if self.strategy_weights else None)
        
        # Handle trial_mixup separately since it needs the label
        if strategy_name == 'trial_mixup':
            if self.trial_cache is not None:
                # Ensure y is not None when same_class_only is True
                if y is None:
                    # For visualization purposes, use 0 as dummy label if none provided
                    return self.trial_mixup(x, 0, same_class_only=False)
                else:
                    return self.trial_mixup(x, y, same_class_only=True)
            else:
                # If no trial cache, return original
                return x.copy()
        
        # Apply the selected strategy if it's in all_methods
        if strategy_name in self.all_methods and self.all_methods[strategy_name] is not None:
            return self.all_methods[strategy_name](x)
        
        # Fallback to original data if strategy not found
        return x.copy()

class StrategyTester:
    """
    Utility class to test different augmentation strategies individually
    or in combinations.
    """
    def __init__(self, visualization_mode=True, augmentation_factor=10):
        # Create the base augmenter with all strategies
        # Enable visualization mode for stronger effects when plotting
        self.base_augmenter = SingleSubjectAugmenter(
            visualization_mode=visualization_mode,
            augmentation_factor=augmentation_factor
        )
        self.available_strategies = list(self.base_augmenter.all_methods.keys())
        self.visualization_mode = visualization_mode
        self.augmentation_factor = augmentation_factor
    
    def get_single_strategy_augmenter(self, strategy_name):
        """Get an augmenter that uses only a single strategy"""
        dprint(f"Creating single strategy augmenter for {strategy_name} with augmentation_factor={self.augmentation_factor}")
        return SingleSubjectAugmenter(
            augmentation_strategies=[strategy_name],
            strategy_weights=[1.0],
            visualization_mode=self.visualization_mode,
            augmentation_factor=self.augmentation_factor
        )
    
    def get_combined_strategy_augmenter(self, strategy_names, strategy_weights=None):
        """Get an augmenter that uses a combination of strategies with optional weights"""
        # Validate strategy names
        for name in strategy_names:
            if name not in self.available_strategies:
                raise ValueError(f"Strategy '{name}' not found in available methods")
        
        dprint(f"Creating combined strategy augmenter for {strategy_names} with augmentation_factor={self.augmentation_factor}")
        # Create a new augmenter with the specified strategies
        return SingleSubjectAugmenter(
            augmentation_strategies=strategy_names,
            strategy_weights=strategy_weights,
            visualization_mode=self.visualization_mode,
            augmentation_factor=self.augmentation_factor
        )
    
    def get_all_single_strategy_augmenters(self):
        """Get a dictionary of augmenters, one for each available strategy"""
        return {
            strategy: self.get_single_strategy_augmenter(strategy)
            for strategy in self.available_strategies
        }
        
    def visualize_augmentation(self, X, strategy_name=None, n_samples=3, n_augmentations=5):
        """
        Visualize the effect of a specific augmentation strategy
        
        Args:
            X: Input data [n_samples, n_channels, n_times]
            strategy_name: Name of the strategy to visualize (None for combined)
            n_samples: Number of original samples to visualize
            n_augmentations: Number of augmentations to create for each sample
            
        Returns:
            Figure object that can be displayed
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        # Select random samples
        if X.shape[0] <= n_samples:
            sample_indices = np.arange(X.shape[0])
        else:
            sample_indices = np.random.choice(X.shape[0], n_samples, replace=False)
        
        # Create a figure
        fig = plt.figure(figsize=(15, n_samples * 3))
        gs = GridSpec(n_samples, n_augmentations + 1, figure=fig)
        
        # Get the appropriate augmenter
        if strategy_name is None:
            augmenter = self.base_augmenter
            title = "Combined Augmentation"
        else:
            augmenter = self.get_single_strategy_augmenter(strategy_name)
            title = f"Strategy: {strategy_name}"
        
        # Initialize trial cache for mixup if needed
        if strategy_name == 'trial_mixup' or (strategy_name is None and 'trial_mixup' in self.base_augmenter.active_strategies):
            # Create dummy labels if none provided (just for visualization)
            dummy_labels = np.zeros(X.shape[0], dtype=int)
            augmenter.trial_cache = {'data': X, 'labels': dummy_labels}
        
        # For each sample
        for i, idx in enumerate(sample_indices):
            # Plot original
            ax = fig.add_subplot(gs[i, 0])
            self._plot_eeg(ax, X[idx], title="Original" if i == 0 else None)
            
            # Plot augmentations
            for j in range(n_augmentations):
                ax = fig.add_subplot(gs[i, j+1])
                
                # Apply augmentation
                if strategy_name is None:
                    augmented = augmenter.augment(X[idx])
                else:
                    # For trial_mixup, we need to pass a label
                    if strategy_name == 'trial_mixup':
                        augmented = augmenter.apply_single_augmentation(X[idx], y=0, strategy_name=strategy_name)
                    else:
                        augmented = augmenter.apply_single_augmentation(X[idx], strategy_name=strategy_name)
                
                self._plot_eeg(ax, augmented, title=f"Aug {j+1}" if i == 0 else None)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        return fig
    
    def _plot_eeg(self, ax, eeg_data, title=None):
        """Helper to plot EEG data"""
        n_channels, n_times = eeg_data.shape
        
        # Offset each channel for visibility
        offsets = np.arange(n_channels) * 2
        
        for c in range(n_channels):
            ax.plot(eeg_data[c] + offsets[c], linewidth=0.8)
        
        # Remove ticks for clarity
        ax.set_yticks([])
        ax.set_xticks([])
        
        if title:
            ax.set_title(title)
