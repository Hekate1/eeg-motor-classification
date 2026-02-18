"""
Single Subject EEG Classification with Advanced Techniques

This script focuses on classifying a single subject's EEG motor imagery data with:
1. Enhanced data augmentation
2. Multiple model architectures (EEGNet, 1D Transformer, LSTM)
3. Pre-training on multi-subject data followed by fine-tuning

Requires eeg_classification.py and enhanced_preprocessing.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import mne

# Import from existing modules
from eeg_classification import (
    load_processed_data,
    extract_features_and_labels,
    enhanced_preprocessing,
    extract_spectral_features,
    run_classification,
    EnhancedEEGNet,
    EnhancedEEGNetClassifier,
    select_motor_channels,
    SubjectAdaptationLayer,
    plot_results
)

# Global configuration
SUBJECT_ID = "001"
RUN_IDS = ["3", "7", "11"]  # Motor imagery runs
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_subject_data(subject_id=SUBJECT_ID, run_ids=RUN_IDS, use_motor_channels=True):
    """Load data for a single subject with optional channel selection and preprocessing"""
    try:
        epochs = load_processed_data(subject_id, run_ids)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Creating sample data for demonstration purposes...")
        # Create sample data directory
        os.makedirs("processed_data", exist_ok=True)
        
        # Create sample data with typical dimensions for motor imagery
        n_channels = 64
        n_times = 640  # 4 seconds at 160Hz
        n_trials = 100  # 50 per class
        
        # Generate random EEG-like data
        np.random.seed(RANDOM_SEED)
        X = np.random.randn(n_trials, n_channels, n_times) * 0.5
        
        # Create labels (50 trials per class for binary classification)
        y = np.concatenate([np.zeros(n_trials//2), np.ones(n_trials//2)])
        
        # Create info for MNE epochs
        ch_names = [f'EEG{i:03d}' for i in range(1, n_channels+1)]
        ch_types = ['eeg'] * n_channels
        info = mne.create_info(ch_names=ch_names, sfreq=160, ch_types=ch_types)
        
        # Create events array (required for MNE epochs)
        events = np.zeros((n_trials, 3), dtype=int)
        events[:, 0] = np.arange(0, n_trials * 100, 100)  # Event times
        events[:, 2] = y + 1  # Event IDs (1 or 2)
        
        # Create MNE epochs with the generated data
        epochs = mne.EpochsArray(X, info, events=events, tmin=0)
        epochs.event_id = {'TASK1T1': 1, 'TASK1T2': 2}  # Standard motor imagery event IDs
        
        print(f"Created sample data with {n_trials} trials, {n_channels} channels, and {n_times} timepoints")
        
        # Skip filtering to avoid circular imports
        X, y = extract_features_and_labels(epochs, tmin=0.5, tmax=3.0)
        X_spectral = extract_spectral_features(X, sfreq=160)
        return X, y, X_spectral, epochs.info
    
    if use_motor_channels:
        original_channels = len(epochs.ch_names)
        epochs = select_motor_channels(epochs)
        print(f"Selected {len(epochs.ch_names)} motor-related channels out of {original_channels} total channels")
    
    # Apply mu/beta band filtering (optimal for motor imagery)
    epochs_filtered = enhanced_preprocessing(epochs, fmin=8, fmax=30)
    
    # Extract features and labels (with optimal time window for motor imagery)
    X, y = extract_features_and_labels(epochs_filtered, tmin=0.5, tmax=3.0)
    X_spectral = extract_spectral_features(X, sfreq=epochs.info['sfreq'])
    
    return X, y, X_spectral, epochs.info


class AdvancedEEGDataAugmenter:
    """Advanced data augmentation techniques specifically for EEG motor imagery data"""
    
    def __init__(self, 
                 noise_level=0.05,
                 time_shift_percent=0.2,
                 channel_mask_prob=0.2, 
                 freq_mask_prob=0.3,
                 scaling_prob=0.3,
                 scaling_range=(0.8, 1.2)):
        """Initialize the augmenter with configuration parameters
        
        Parameters:
        -----------
        noise_level : float
            Maximum standard deviation of Gaussian noise to add
        time_shift_percent : float
            Maximum percentage of signal length to shift
        channel_mask_prob : float
            Probability of masking each channel
        freq_mask_prob : float
            Probability of applying frequency masking
        scaling_prob : float
            Probability of applying amplitude scaling
        scaling_range : tuple
            Range of scaling factors (min, max)
        """
        self.noise_level = noise_level
        self.time_shift_percent = time_shift_percent
        self.channel_mask_prob = channel_mask_prob
        self.freq_mask_prob = freq_mask_prob
        self.scaling_prob = scaling_prob
        self.scaling_range = scaling_range
    
    def augment(self, X, copies=5):
        """Generate augmented copies of the input data
        
        Parameters:
        -----------
        X : ndarray, shape (n_trials, n_channels, n_times)
            EEG data to augment
        copies : int
            Number of augmented copies to generate for each original trial
            
        Returns:
        --------
        ndarray, shape (n_trials * (copies+1), n_channels, n_times)
            Original and augmented data combined
        """
        n_trials, n_channels, n_times = X.shape
        X_aug_list = [X]  # Start with original data
        
        for _ in range(copies):
            X_copy = X.copy()
            
            # Apply random amplitude scaling
            if np.random.random() < self.scaling_prob:
                scaling_factors = np.random.uniform(
                    self.scaling_range[0], 
                    self.scaling_range[1], 
                    size=(n_trials, n_channels, 1)
                )
                X_copy = X_copy * scaling_factors
            
            # Add random Gaussian noise
            noise_amplitude = np.random.uniform(0, self.noise_level)
            X_copy += np.random.normal(0, noise_amplitude, X_copy.shape)
            
            # Apply random time shifts
            for i in range(n_trials):
                if np.random.random() < 0.7:  # 70% chance of time shift
                    max_shift = int(n_times * self.time_shift_percent)
                    shift = np.random.randint(-max_shift, max_shift + 1)
                    if shift > 0:
                        X_copy[i, :, shift:] = X_copy[i, :, :-shift]
                        X_copy[i, :, :shift] = 0
                    elif shift < 0:
                        X_copy[i, :, :shift] = X_copy[i, :, -shift:]
                        X_copy[i, :, shift:] = 0
            
            # Apply channel masking
            for i in range(n_trials):
                mask = np.random.random(n_channels) < self.channel_mask_prob
                if mask.sum() > 0:  # Ensure at least one channel is masked
                    X_copy[i, mask, :] = 0
            
            # Apply frequency domain masking
            if np.random.random() < self.freq_mask_prob:
                for i in range(n_trials):
                    for j in range(n_channels):
                        if np.random.random() < 0.3:  # Apply to 30% of channels
                            # FFT transform
                            fft = np.fft.rfft(X_copy[i, j, :])
                            # Create random frequency mask
                            mask = np.ones(len(fft), dtype=bool)
                            mask_start = np.random.randint(0, len(fft) // 3)
                            mask_width = np.random.randint(1, len(fft) // 10)
                            mask_end = min(mask_start + mask_width, len(mask))
                            mask[mask_start:mask_end] = False
                            # Apply mask and inverse FFT
                            fft[~mask] = 0
                            X_copy[i, j, :] = np.fft.irfft(fft, n=n_times)
            
            X_aug_list.append(X_copy)
        
        # Combine original and augmented data
        X_augmented = np.vstack(X_aug_list)
        
        return X_augmented
    
    def augment_with_labels(self, X, y, copies=5):
        """Generate augmented copies of data with corresponding labels
        
        Parameters:
        -----------
        X : ndarray, shape (n_trials, n_channels, n_times)
            EEG data to augment
        y : ndarray, shape (n_trials,)
            Labels for each trial
        copies : int
            Number of augmented copies to generate for each original trial
            
        Returns:
        --------
        X_aug : ndarray, shape (n_trials * (copies+1), n_channels, n_times)
            Original and augmented data combined
        y_aug : ndarray, shape (n_trials * (copies+1),)
            Labels repeated for each copy
        """
        X_aug = self.augment(X, copies)
        # Repeat labels for each copy
        y_aug = np.tile(y, copies + 1)
        return X_aug, y_aug


class EEGCNN_LSTM(nn.Module):
    """CNN-LSTM hybrid model for EEG classification
    
    Combines convolutional layers for spatial and temporal feature extraction
    with an LSTM layer for sequential dynamics.
    """
    
    def __init__(self, n_channels, n_times, n_classes, dropout_rate=0.5):
        super(EEGCNN_LSTM, self).__init__()
        
        # Convolutional feature extraction
        self.conv_block = nn.Sequential(
            # Temporal convolution
            nn.Conv2d(1, 16, kernel_size=(1, 64), padding='same'),
            nn.BatchNorm2d(16),
            nn.ELU(),
            
            # Spatial convolution
            nn.Conv2d(16, 32, kernel_size=(n_channels, 1)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d((1, 4)),
            nn.Dropout(dropout_rate)
        )
        
        # Calculate feature dimension after convolution
        self.feature_dim = 32 * ((n_times // 4))
        
        # Reshape to sequence for LSTM
        self.sequence_len = n_times // 4
        self.features_per_step = 32
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.features_per_step,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(64 * 2, 32),  # 2 for bidirectional
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, n_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, n_channels, n_times)
        batch_size = x.size(0)
        
        # Add channel dimension for 2D convolution
        x = x.unsqueeze(1)  # (batch_size, 1, n_channels, n_times)
        
        # Apply convolutional block
        x = self.conv_block(x)  # (batch_size, 32, 1, n_times/4)
        
        # Reshape for LSTM: (batch_size, sequence_len, features_per_step)
        x = x.view(batch_size, self.sequence_len, self.features_per_step)
        
        # Apply LSTM
        lstm_out, _ = self.lstm(x)
        
        # Use last time step for classification
        lstm_out = lstm_out[:, -1, :]  # (batch_size, 2*hidden_size)
        
        # Classification
        output = self.classifier(lstm_out)
        
        return output


def create_dataloaders(X, y, X_spectral=None, batch_size=32, val_split=0.2, use_spectral=True):
    """Create training and validation dataloaders
    
    Parameters:
    -----------
    X : ndarray, shape (n_trials, n_channels, n_times)
        EEG data
    y : ndarray, shape (n_trials,)
        Labels
    X_spectral : ndarray, shape (n_trials, n_channels, n_bands)
        Spectral features (optional)
    batch_size : int
        Batch size for training
    val_split : float
        Fraction of data to use for validation
    use_spectral : bool
        Whether to use spectral features
    
    Returns:
    --------
    train_loader : DataLoader
        DataLoader for training data
    val_loader : DataLoader
        DataLoader for validation data
    """
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    # Create dataset with or without spectral features
    if use_spectral and X_spectral is not None:
        X_spectral_tensor = torch.FloatTensor(X_spectral)
        dataset = TensorDataset(X_tensor, X_spectral_tensor, y_tensor)
    else:
        dataset = TensorDataset(X_tensor, y_tensor)
    
    # Split into training and validation sets
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def pretrain_finetune_eegnet(single_subject_id=SUBJECT_ID, all_subject_ids=None, run_ids=RUN_IDS):
    """Pretrain on multiple subjects then finetune on a single subject
    
    Parameters:
    -----------
    single_subject_id : str
        ID of the subject to finetune on
    all_subject_ids : list or None
        List of subject IDs to pretrain on. If None, uses subjects 001-109 except single_subject_id
    run_ids : list
        List of run IDs to use
    
    Returns:
    --------
    dict
        Results and trained model
    """
    if all_subject_ids is None:
        # Use all subjects except the target subject
        all_subject_ids = [f"{i:03d}" for i in range(1, 110) if f"{i:03d}" != single_subject_id]
    
    print(f"Pretraining on {len(all_subject_ids)} subjects, then finetuning on subject {single_subject_id}")
    
    # Step 1: Pretrain on all subjects
    print("\n=== PRETRAINING PHASE ===")
    pretrain_results = run_classification(
        all_subject_ids, run_ids, 
        strategy='enhanced_eegnet',
        tmin=0.5, tmax=3.0,
        preprocessing_params={'fmin': 8, 'fmax': 30},
        dropout_rate=0.5,
        kernel_length=32,
        F1=16, D=4, F2=32,
        lr=0.0003,
        batch_size=64,
        n_epochs=50,  # Fewer epochs for pretraining
        early_stopping_patience=10,
        use_motor_channels=True,
        multi_subject=True
    )
    
    pretrained_model = pretrain_results['classifier'].model
    
    # Step 2: Load single subject data
    print("\n=== FINETUNING PHASE ===")
    X, y, X_spectral, _ = load_subject_data(single_subject_id, run_ids)
    
    # Step 3: Apply data augmentation to single subject data
    print("Applying data augmentation...")
    augmenter = AdvancedEEGDataAugmenter()
    X_aug, y_aug = augmenter.augment_with_labels(X, y, copies=5)
    
    print(f"Data size after augmentation: {X_aug.shape[0]} trials")
    
    # Also augment spectral features if needed
    if X_spectral is not None:
        X_spectral_aug = np.tile(X_spectral, (6, 1, 1))  # Simple duplication for spectral features
    else:
        X_spectral_aug = None
    
    # Step 4: Create a new classifier with the pretrained model weights
    n_classes = len(np.unique(y))
    finetuned_classifier = EnhancedEEGNetClassifier(
        n_classes=n_classes,
        dropout_rate=0.4,  # Lower dropout for finetuning
        kernel_length=32,
        F1=16, D=4, F2=32,
        lr=0.0001,  # Lower learning rate for finetuning
        batch_size=32,
        n_epochs=200,
        weight_decay=0.01,
        augmentation_level=0.3,  # Lower augmentation since we already augmented
        use_spectral=True,
        use_adaptation=True,
        label_smoothing=0.1,
        device=DEVICE
    )
    
    # Initialize with pretrained weights
    finetuned_classifier.model = pretrained_model
    
    # Step 5: Finetune the model
    results = finetuned_classifier.train_and_evaluate(
        X_aug, y_aug, 
        X_spectral=X_spectral_aug,
        val_split=0.2,
        early_stopping_patience=20
    )
    
    # Step 6: Evaluate on original (non-augmented) data
    print("\n=== FINAL EVALUATION ===")
    X_tensor = torch.FloatTensor(X).to(DEVICE)
    y_tensor = torch.LongTensor(y).to(DEVICE)
    
    finetuned_classifier.model.eval()
    with torch.no_grad():
        if X_spectral is not None and finetuned_classifier.use_spectral:
            X_spectral_tensor = torch.FloatTensor(X_spectral).to(DEVICE)
            outputs = finetuned_classifier.model(X_tensor, X_spectral_tensor)
        else:
            outputs = finetuned_classifier.model(X_tensor)
        
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_tensor).float().mean().item()
    
    print(f"Final accuracy on non-augmented data: {accuracy:.4f}")
    
    # Plot confusion matrix
    cm = confusion_matrix(y_tensor.cpu().numpy(), predicted.cpu().numpy())
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - Subject {single_subject_id}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return {
        'classifier': finetuned_classifier,
        'accuracy': accuracy,
        'history': results['history'],
        'confusion_matrix': cm
    }


def train_single_subject_eegnet(subject_id=SUBJECT_ID, run_ids=RUN_IDS):
    """Train EEGNet model on a single subject with augmentation
    
    Parameters:
    -----------
    subject_id : str
        ID of the subject to train on
    run_ids : list
        List of run IDs to use
    
    Returns:
    --------
    dict
        Results and trained model
    """
    # Load data
    X, y, X_spectral, _ = load_subject_data(subject_id, run_ids)
    
    # Apply data augmentation
    print("Applying data augmentation...")
    augmenter = AdvancedEEGDataAugmenter()
    X_aug, y_aug = augmenter.augment_with_labels(X, y, copies=5)
    
    print(f"Data size after augmentation: {X_aug.shape[0]} trials")
    
    # Also augment spectral features
    if X_spectral is not None:
        X_spectral_aug = np.tile(X_spectral, (6, 1, 1))  # Simple duplication for spectral features
    else:
        X_spectral_aug = None
    
    # Run classification with enhanced EEGNet
    results = run_classification(
        [subject_id], run_ids,  # Only this subject
        strategy='enhanced_eegnet',
        X=X_aug, y=y_aug, X_spectral=X_spectral_aug,  # Use augmented data
        dropout_rate=0.6,
        kernel_length=32,
        F1=16, D=4, F2=32,
        lr=0.0003,
        batch_size=32,
        n_epochs=300,
        weight_decay=0.01,
        early_stopping_patience=40,
        augmentation_level=0.3,  # Lower since we already augmented
        use_spectral=True,
        use_adaptation=True,
        label_smoothing=0.1,
        multi_subject=False
    )
    
    # Plot results
    plot_results(results, 'enhanced_eegnet')
    
    return results


def train_single_subject_lstm(subject_id=SUBJECT_ID, run_ids=RUN_IDS):
    """Train CNN-LSTM model on a single subject with augmentation
    
    Parameters:
    -----------
    subject_id : str
        ID of the subject to train on
    run_ids : list
        List of run IDs to use
    
    Returns:
    --------
    dict
        Results and trained model
    """
    # Load data
    X, y, _, _ = load_subject_data(subject_id, run_ids)
    
    # Apply data augmentation
    print("Applying data augmentation...")
    augmenter = AdvancedEEGDataAugmenter()
    X_aug, y_aug = augmenter.augment_with_labels(X, y, copies=5)
    
    print(f"Data size after augmentation: {X_aug.shape[0]} trials")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        X_aug, y_aug, batch_size=32, val_split=0.2, use_spectral=False
    )
    
    # Initialize model
    n_channels, n_times = X.shape[1], X.shape[2]
    n_classes = len(np.unique(y))
    
    model = EEGCNN_LSTM(n_channels, n_times, n_classes, dropout_rate=0.5).to(DEVICE)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, verbose=True
    )
    
    # Training parameters
    n_epochs = 200
    early_stopping_patience = 30
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Training loop
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Print progress
        print(f'Epoch {epoch+1}/{n_epochs}: '
              f'train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, '
              f'val_loss={val_loss:.4f}, val_acc={val_acc:.4f}')
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation on original (non-augmented) data
    X_tensor = torch.FloatTensor(X).to(DEVICE)
    y_tensor = torch.LongTensor(y).to(DEVICE)
    
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_tensor).float().mean().item()
    
    print(f"Final accuracy on non-augmented data: {accuracy:.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], 'b-', label='Training Loss')
    plt.plot(history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss (CNN-LSTM)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(history['val_acc'], 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy (CNN-LSTM)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Plot confusion matrix
    cm = confusion_matrix(y_tensor.cpu().numpy(), predicted.cpu().numpy())
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - Subject {subject_id}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return {
        'model': model,
        'accuracy': accuracy,
        'history': history,
        'confusion_matrix': cm
    }


def run_all_models_comparison(subject_id=SUBJECT_ID, run_ids=RUN_IDS, models=None):
    """Run all models or specific models on a single subject and compare results
    
    Parameters:
    -----------
    subject_id : str
        ID of the subject to train on
    run_ids : list
        List of run IDs to use
    models : list or None
        List of model names to run ['eegnet', 'cnn_lstm', 'pretrained_eegnet']
        If None, run all models
        
    Returns:
    --------
    dict
        Results for all models
    """
    results = {}
    
    # Default to running all models if none specified
    if models is None:
        models = ['eegnet', 'cnn_lstm', 'pretrained_eegnet']
    
    # Run specified models
    if 'eegnet' in models:
        print("\n=== TRAINING STANDALONE EEGNET ===")
        results['eegnet'] = train_single_subject_eegnet(subject_id, run_ids)
    
    if 'cnn_lstm' in models:
        print("\n=== TRAINING CNN-LSTM MODEL ===")
        results['cnn_lstm'] = train_single_subject_lstm(subject_id, run_ids)
    
    if 'pretrained_eegnet' in models:
        print("\n=== TRAINING PRE-TRAINED THEN FINE-TUNED EEGNET ===")
        results['pretrained_eegnet'] = pretrain_finetune_eegnet(subject_id, run_ids=run_ids)
    
    # Summarize results
    print("\n=== MODEL COMPARISON ===")
    print(f"{'Model':<25} {'Accuracy':<10}")
    print("-" * 35)
    
    # Print results for models that were run
    if 'eegnet' in results:
        eegnet_acc = results['eegnet'].get('final_accuracy', 
                    results['eegnet'].get('accuracy', 0.0))
        print(f"{'EEGNet':<25} {eegnet_acc:.4f}")
    
    if 'cnn_lstm' in results:
        cnnlstm_acc = results['cnn_lstm'].get('accuracy', 0.0)
        print(f"{'CNN-LSTM':<25} {cnnlstm_acc:.4f}")
    
    if 'pretrained_eegnet' in results:
        pretrained_acc = results['pretrained_eegnet'].get('accuracy', 0.0)
        print(f"{'Pretrained+Finetuned EEGNet':<25} {pretrained_acc:.4f}")
    
    return results


if __name__ == "__main__":
    # Run a specific model for faster testing
    # Uncomment one of these lines:
    # results = run_all_models_comparison(models=['eegnet'])  # EEGNet only
    # results = run_all_models_comparison(models=['cnn_lstm'])  # CNN-LSTM only
    # results = run_all_models_comparison(models=['pretrained_eegnet'])  # Pretrained EEGNet only
    
    # Or run all models
    results = run_all_models_comparison() 