"""
Data augmentation module for EEG classification.

This module provides classes and functions for augmenting EEG data
to improve model generalization.
"""

import numpy as np


class EEGDataAugmenter:
    """
    Class for applying data augmentation techniques to EEG data
    """
    def __init__(self, noise_level=0.03, scale_range=(0.9, 1.1),
                 chunk_max_offset=30, support_variable_length=False):
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
        self.scale_min, self.scale_max = scale_range
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
        scale = np.random.uniform(self.scale_min, self.scale_max)
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
        # Core augmentations - reliable and effective
        augmentations = [
            self.add_noise,
            self.scale_amplitude,
            self.spectral_perturbation
        ]
        
        # Apply 1-2 random augmentations
        num_augmentations = np.random.randint(1, 3)
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
    
    def augment_batch(self, X, y, augmentation_factor=2):
        """
        Generate augmented data for a batch of EEG trials
        
        Args:
            X: EEG data with shape (n_trials, n_channels, n_times)
            y: Labels with shape (n_trials)
            augmentation_factor: Number of augmented copies to generate per original
            
        Returns:
            X_augmented: Augmented data
            y_augmented: Corresponding labels
        """
        n_trials = X.shape[0]
        
        # Cache a copy of the original trials for mixup augmentation
        self.trial_cache = [X[i].copy() for i in range(n_trials)]
        if y is not None:
            self.trial_labels = y.copy()
        
        # Start with the original data
        augmented_X = [X]
        augmented_y = [y]
        
        # Generate augmented data
        for _ in range(augmentation_factor):
            # Generate augmented version of each trial
            X_batch_aug = np.array([self.augment(X[i], y[i] if y is not None else None) 
                                  for i in range(n_trials)])
            
            # Add to collection
            augmented_X.append(X_batch_aug)
            augmented_y.append(y)
        
        # Concatenate all data
        X_augmented = np.concatenate(augmented_X, axis=0)
        y_augmented = np.concatenate(augmented_y, axis=0)
        
        print(f"Generated {len(y_augmented)} samples from {n_trials} original samples")
        return X_augmented, y_augmented


class OptimizedAugmenter(EEGDataAugmenter):
    """Optimized augmentation strategy focusing on the most effective methods"""
    
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


class BasicAugmenter(EEGDataAugmenter):
    """Basic augmentation strategy with just noise and amplitude scaling"""
    
    def __init__(self, noise_level=0.03, scale_range=(0.9, 1.1), chunk_max_offset=30):
        super().__init__(noise_level, scale_range, chunk_max_offset, support_variable_length=False)
        
        # Just the simplest methods for maximum reliability
        self.basic_methods = {
            'add_noise': self.add_noise,
            'scale_amplitude': self.scale_amplitude
        }
    
    def augment(self, x, y=None):
        """Apply basic augmentation strategy - just noise and scaling"""
        x_aug = x.copy()
        
        # Add noise
        x_aug = self.add_noise(x_aug)
        
        # Scale amplitude with 50% probability
        if np.random.rand() < 0.5:
            x_aug = self.scale_amplitude(x_aug)
            
        return x_aug 