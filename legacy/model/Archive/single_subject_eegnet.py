"""
Single Subject EEG Classification with EEGNET

This script provides a simplified approach to classifying EEG motor imagery data
for a single subject using the EEGNET architecture.

Structure is designed to be modular and easily extensible for future modifications:
- Adding data augmentation techniques
- Incorporating additional models
- Supporting pretrained/transfer learning approaches
"""

import os
import numpy as np
import mne
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
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
    
    Parameters:
    -----------
    subject_id : str
        Subject ID to load
    run_ids : list
        List of run IDs to load
    use_motor_channels : bool
        Whether to select only motor-related channels
        
    Returns:
    --------
    X : ndarray, shape (n_trials, n_channels, n_times)
        EEG data
    y : ndarray, shape (n_trials,)
        Class labels
    info : dict
        Information about the data
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
    
    # Extract data for motor imagery classification
    # Find left hand (TASK1T1) vs right hand (TASK1T2) events
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
    # This is when motor imagery patterns are typically most pronounced
    tmin, tmax = 0.5, 3.0
    epochs_cropped = epochs_filtered.copy().crop(tmin=tmin, tmax=tmax)
    
    # Extract data for the selected events
    selected_epochs = epochs_cropped[list(motor_events.keys())]
    
    # Get data and labels
    X = selected_epochs.get_data()
    y = selected_epochs.events[:, 2]
    
    # Convert event IDs to zero-indexed class labels
    unique_labels = np.unique(y)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    y = np.array([label_map[label] for label in y])
    
    # Print class distribution
    print(f"Class distribution: {np.bincount(y)} (total: {len(y)} trials)")
    
    return X, y, epochs.info


class EEGNet(nn.Module):
    """
    EEGNet: a compact convolutional neural network for EEG signal processing.
    
    Based on the paper:
    Lawhern et al. "EEGNet: A Compact Convolutional Network for EEG-based
    Brain-Computer Interfaces." Journal of Neural Engineering, 2018.
    
    Architecture consists of:
    1. Temporal convolution to learn frequency filters
    2. Depthwise convolution to learn spatial filters
    3. Separable convolution to learn cross-feature mappings
    4. Classification layer
    """
    
    def __init__(self, n_channels, n_times, n_classes, 
                 dropout_rate=0.5, kernel_length=64, F1=8, D=2, F2=16):
        """
        Initialize the EEGNet model.
        
        Parameters:
        -----------
        n_channels : int
            Number of EEG channels
        n_times : int
            Number of time points
        n_classes : int
            Number of output classes
        dropout_rate : float
            Dropout probability
        kernel_length : int
            Length of temporal convolution kernel
        F1 : int
            Number of temporal filters
        D : int
            Depth multiplier for depthwise convolution
        F2 : int
            Number of separable convolution filters
        """
        super(EEGNet, self).__init__()
        
        # Store parameters
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        
        # Block 1: Temporal Convolution
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kernel_length), padding='same', bias=False),
            nn.BatchNorm2d(F1),
            nn.ELU()
        )
        
        # Block 2: Depthwise Convolution
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout_rate)
        )
        
        # Calculate feature map dimensions after pooling
        self.feature_size_after_pool1 = n_times // 4
        
        # Block 3: Separable Convolution
        self.block3 = nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16), groups=F1 * D, padding='same', bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            # Pointwise convolution
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout_rate)
        )
        
        # Calculate feature map dimensions after second pooling
        self.feature_size_after_pool2 = self.feature_size_after_pool1 // 8
        
        # Ensure feature size is at least 1
        if self.feature_size_after_pool2 < 1:
            print(f"WARNING: Time dimension after pooling is too small: {self.feature_size_after_pool2}")
            self.feature_size_after_pool2 = 1
        
        # Calculate total number of features for classification layer
        self.classifier_input_size = F2 * self.feature_size_after_pool2
        
        # Classification block with more layers and regularization
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_input_size, F2 * 2),
            nn.BatchNorm1d(F2 * 2),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(F2 * 2, F2),
            nn.BatchNorm1d(F2),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(F2, n_classes)
        )

    def forward(self, x):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        x : torch.Tensor, shape (batch_size, n_channels, n_times)
            Input EEG data
            
        Returns:
        --------
        torch.Tensor, shape (batch_size, n_classes)
            Class logits
        """
        # Add channel dimension for Conv2D (batch_size, 1, n_channels, n_times)
        x = x.unsqueeze(1)
        
        # Block 1: Temporal Convolution
        x = self.block1(x)
        
        # Block 2: Depthwise Convolution
        x = self.block2(x)
        
        # Block 3: Separable Convolution
        x = self.block3(x)
        
        # Reshape for classifier (batch_size, features)
        x = x.view(x.size(0), -1)
        
        # Check if shape matches expected shape
        expected_shape = self.classifier_input_size
        if x.shape[1] != expected_shape:
            print(f"WARNING: Feature shape mismatch. Got {x.shape[1]}, expected {expected_shape}")
            # Adaptive average pooling as a fallback
            x = x.view(x.size(0), -1, 1).mean(dim=1)
        
        # Classification
        x = self.classifier(x)
        
        return x


class EEGNetClassifier:
    """
    EEGNet classifier with training, evaluation, and prediction functionality.
    """
    
    def __init__(self, n_classes, dropout_rate=0.5, kernel_length=64, 
                 F1=8, D=2, F2=16, lr=0.001, batch_size=32, 
                 n_epochs=100, weight_decay=0.01, device=None):
        """
        Initialize the EEGNet classifier.
        
        Parameters:
        -----------
        n_classes : int
            Number of output classes
        dropout_rate : float
            Dropout probability
        kernel_length : int
            Length of temporal convolution kernel
        F1 : int
            Number of temporal filters
        D : int
            Depth multiplier for depthwise convolution
        F2 : int
            Number of separable convolution filters
        lr : float
            Learning rate
        batch_size : int
            Batch size for training
        n_epochs : int
            Number of training epochs
        weight_decay : float
            L2 regularization strength
        device : torch.device or None
            Device to use for training (defaults to CUDA if available)
        """
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        self.kernel_length = kernel_length
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.weight_decay = weight_decay
        self.device = device if device is not None else DEVICE
        
        # Model will be initialized when data dimensions are known
        self.model = None
        
        print(f"Using device: {self.device}")
    
    def _prepare_data(self, X, y, val_split=0.2):
        """
        Prepare data for training by creating dataloaders.
        
        Parameters:
        -----------
        X : ndarray, shape (n_trials, n_channels, n_times)
            EEG data
        y : ndarray, shape (n_trials,)
            Class labels
        val_split : float
            Fraction of data to use for validation
            
        Returns:
        --------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader
            Validation data loader
        """
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        # Create dataset
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Split into training and validation sets
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], 
            generator=torch.Generator().manual_seed(RANDOM_SEED)
        )
        
        # Create dataloaders with balanced sampling
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
    
    def train_and_evaluate(self, X, y, val_split=0.2, early_stopping_patience=20):
        """
        Train and evaluate the EEGNet model.
        
        Parameters:
        -----------
        X : ndarray, shape (n_trials, n_channels, n_times)
            EEG data
        y : ndarray, shape (n_trials,)
            Class labels
        val_split : float
            Fraction of data to use for validation
        early_stopping_patience : int
            Number of epochs to wait for improvement before stopping
            
        Returns:
        --------
        dict
            Training history and evaluation metrics
        """
        n_channels, n_times = X.shape[1], X.shape[2]
        
        # Initialize model
        self.model = EEGNet(
            n_channels=n_channels,
            n_times=n_times,
            n_classes=self.n_classes,
            dropout_rate=self.dropout_rate,
            kernel_length=self.kernel_length,
            F1=self.F1,
            D=self.D,
            F2=self.F2
        ).to(self.device)
        
        # Prepare data
        train_loader, val_loader = self._prepare_data(X, y, val_split)
        
        # Loss function with label smoothing
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Optimizer with gradient clipping
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler with warmup
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            epochs=self.n_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1
        )
        
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
        for epoch in range(self.n_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                # Track loss and accuracy
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += targets.size(0)
                train_correct += (predicted == targets).sum().item()
            
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
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
                
                val_loss = val_loss / val_total
                val_acc = val_correct / val_total
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            # Print progress
            print(f'Epoch {epoch+1}/{self.n_epochs}: '
                  f'train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, '
                  f'val_loss={val_loss:.4f}, val_acc={val_acc:.4f}')
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # Final evaluation
        self.model.eval()
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        cm = confusion_matrix(all_targets, all_predictions)
        
        print(f'\nFinal validation accuracy: {accuracy:.4f}')
        print('\nConfusion Matrix:')
        print(cm)
        
        return {
            'model': self.model,
            'history': history,
            'accuracy': accuracy,
            'confusion_matrix': cm
        }
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        X : ndarray, shape (n_trials, n_channels, n_times)
            EEG data
            
        Returns:
        --------
        ndarray, shape (n_trials,)
            Predicted class labels
        """
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


def train_single_subject_eegnet(subject_id=SUBJECT_ID, run_ids=RUN_IDS):
    """
    Train an EEGNet model on a single subject's data.
    
    Parameters:
    -----------
    subject_id : str
        Subject ID to train on
    run_ids : list
        List of run IDs to use
        
    Returns:
    --------
    dict
        Training results
    """
    # Load and preprocess data
    X, y, info = load_subject_data(subject_id, run_ids)
    
    # Print dataset information
    print(f"\nTraining EEGNet for subject {subject_id}")
    print(f"Data shape: {X.shape} (trials, channels, time points)")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Create and train classifier with modified parameters
    classifier = EEGNetClassifier(
        n_classes=len(np.unique(y)),
        dropout_rate=0.3,      # Reduced dropout
        kernel_length=32,      # Smaller kernel size
        F1=16,                # More filters
        D=2,                  # Same depth multiplier
        F2=32,                # More separable filters
        lr=0.0001,            # Much lower learning rate
        batch_size=8,         # Smaller batch size
        n_epochs=300,         # More epochs
        weight_decay=0.0001   # Reduced weight decay
    )
    
    # Train and evaluate
    results = classifier.train_and_evaluate(X, y, val_split=0.2, early_stopping_patience=40)
    
    # Add the trained classifier to the results
    results['classifier'] = classifier
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(results['history']['train_loss'], 'b-', label='Training Loss')
    plt.plot(results['history']['val_loss'], 'r-', label='Validation Loss')
    plt.title(f'Training and Validation Loss - Subject {subject_id}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(results['history']['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(results['history']['val_acc'], 'r-', label='Validation Accuracy')
    plt.title(f'Training and Validation Accuracy - Subject {subject_id}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - Subject {subject_id}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return results


if __name__ == "__main__":
    # Set the subject ID and run IDs
    subject_id = SUBJECT_ID
    run_ids = RUN_IDS
    
    # Train EEGNet on the specified subject
    results = train_single_subject_eegnet(subject_id, run_ids)
    
    print(f"\nFinal accuracy: {results['accuracy']:.4f}")
    
    """
    Extension Points for Future Development:
    
    1. Data Augmentation:
       - Add a data augmentation class with methods for different techniques
       - Include time shifts, noise addition, and channel masking
       - Implement in a way that can be easily enabled/disabled
    
    2. Additional Models:
       - Create a base model class that EEGNet and future models can inherit from
       - Implement LSTM, Transformer, or hybrid models following the same interface
       - Add a model factory function to select which model to use
    
    3. Transfer Learning / Pretraining:
       - Modify the training function to support pretrained weights
       - Add two-phase training (pretrain on multiple subjects, finetune on target)
       - Implement subject adaptation techniques
    """ 