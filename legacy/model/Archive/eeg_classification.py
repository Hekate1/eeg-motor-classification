import os
import numpy as np
import mne
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mne.decoding import CSP
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
import torch.nn.functional as F
import scipy

def load_processed_data(subject_id, run_ids):
    """Load processed epochs data for a specific subject and multiple runs.
    
    Parameters:
    -----------
    subject_id : str or list
        Subject ID (if list, only the first subject will be used)
    run_ids : list of str
        List of run IDs to load and combine
    
    Returns:
    --------
    mne.Epochs
        Combined epochs from all specified runs
    """
    all_epochs = []
    
    # Handle case where subject_id is a list
    if isinstance(subject_id, list):
        subject_id = subject_id[0]
    
    for run_id in run_ids:
        filepath = f'processed_data/sub-{subject_id}_run-{run_id}_processed-epo.fif'
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No processed data found at {filepath}")
        
        epochs = mne.read_epochs(filepath)
        all_epochs.append(epochs)
    
    # Combine epochs from all runs
    combined_epochs = mne.concatenate_epochs(all_epochs)
    print(f"Loaded {len(combined_epochs)} total epochs from {len(run_ids)} runs")
    return combined_epochs

def extract_features_and_labels(epochs, event_ids=None, tmin=0.0, tmax=4.0):
    """Extract data and labels from epochs for classification.
    
    Parameters:
    -----------
    epochs : mne.Epochs
        The epochs object containing the data
    event_ids : dict or None
        Dictionary mapping event names to IDs. If None, use left vs right hand (TASK1T1 vs TASK1T2).
    tmin, tmax : float
        Start and end time for epochs to use for classification
    
    Returns:
    --------
    X : ndarray, shape (n_epochs, n_channels, n_times)
        The data for classification
    y : ndarray, shape (n_epochs,)
        The labels for classification
    """
    if event_ids is None:
        # Explicitly use left hand (TASK1T1) vs right hand (TASK1T2)
        motor_events = {}
        if 'TASK1T1' in epochs.event_id:
            motor_events['TASK1T1'] = epochs.event_id['TASK1T1']  # Left hand
        if 'TASK1T2' in epochs.event_id:
            motor_events['TASK1T2'] = epochs.event_id['TASK1T2']  # Right hand
        
        if len(motor_events) < 2:
            raise ValueError("Could not find both left (TASK1T1) and right (TASK1T2) hand events in the data")
        
        event_ids = motor_events
    
    print(f"Classifying between event types: {list(event_ids.keys())}")
    
    # Crop epochs to the specified time range
    epochs_cropped = epochs.copy().crop(tmin=tmin, tmax=tmax)
    
    # Extract data for the selected events
    selected_epochs = epochs_cropped[list(event_ids.keys())]
    
    # Get data and labels
    X = selected_epochs.get_data()
    y = selected_epochs.events[:, 2]
    
    # Adjust labels to be zero-indexed (for most classifiers)
    unique_labels = np.unique(y)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    y_mapped = np.array([label_map[label] for label in y])
    
    # Print class distribution
    print(f"Class distribution: {np.bincount(y_mapped)} (total: {len(y_mapped)} trials)")
    
    return X, y_mapped

def enhanced_preprocessing(epochs, fmin=8, fmax=30):
    """Apply enhanced preprocessing optimized for motor imagery
    
    Parameters:
    -----------
    epochs : mne.Epochs
        The epochs to process
    fmin, fmax : float
        Frequency band to filter data (default: mu+beta band 8-30Hz)
    
    Returns:
    --------
    mne.Epochs
        The processed epochs
    """
    # Use FIR filter with zero-phase (instead of IIR) to avoid phase distortion
    # hamming window and proper filter length gives good frequency response
    epochs_filtered = epochs.copy().filter(
        l_freq=fmin, h_freq=fmax, 
        method='fir',
        fir_window='hamming', 
        fir_design='firwin',
        verbose=False
    )
    
    print(f"Applied FIR filter {fmin}-{fmax}Hz (zero phase)")
    return epochs_filtered

def extract_spectral_features(X, sfreq=160):
    """Extract band power features using Welch method
    
    Parameters:
    -----------
    X : ndarray, shape (n_epochs, n_channels, n_times)
        EEG data
    sfreq : float
        Sampling frequency of the data
    
    Returns:
    --------
    ndarray, shape (n_epochs, n_channels, n_bands)
        Extracted band powers
    """
    # Define frequency bands of interest
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),  # mu rhythm
        'beta': (13, 30),
        'gamma': (30, 45)
    }
    
    n_epochs, n_channels, n_times = X.shape
    n_bands = len(bands)
    
    # Initialize array for band powers
    X_freq = np.zeros((n_epochs, n_channels, n_bands))
    
    # Calculate optimal segment length for Welch
    nperseg = min(128, n_times // 2)
    
    # Extract band powers for each epoch and channel
    for i in range(n_epochs):
        for j in range(n_channels):
            # Use Welch method to estimate power spectral density
            f, psd = scipy.signal.welch(X[i, j], fs=sfreq, nperseg=nperseg)
            
            # Extract power in each frequency band
            for k, (band_name, (fmin, fmax)) in enumerate(bands.items()):
                freq_mask = (f >= fmin) & (f <= fmax)
                X_freq[i, j, k] = np.sum(psd[freq_mask])
    
    print(f"Extracted spectral features ({n_bands} frequency bands)")
    return X_freq

class CSPSVMClassifier:
    """CSP followed by SVM classifier for EEG data."""
    
    def __init__(self, n_components=4, kernel='linear', C=1.0, pca_components=None):
        """Initialize the CSP + SVM pipeline.
        
        Parameters:
        -----------
        n_components : int
            Number of CSP components to use
        kernel : str
            Kernel type for SVM ('linear', 'rbf', etc.)
        C : float
            Regularization parameter for SVM
        pca_components : int or None
            Number of PCA components to use before CSP. If None, PCA is not applied.
            If 'auto', the number of components will be determined automatically.
        """
        self.n_components = n_components
        self.kernel = kernel
        self.C = C
        self.pca_components = pca_components
        self.pipeline = None  # Will be initialized in train_and_evaluate
    
    def train_and_evaluate(self, X, y, cv=5, random_state=42):
        """Train and evaluate the classifier using cross-validation.
        
        Parameters:
        -----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
            The EEG data
        y : ndarray, shape (n_epochs,)
            The class labels
        cv : int
            Number of cross-validation folds
        random_state : int
            Random seed for reproducibility
        
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        # Custom transformer for reshaping 3D EEG data for PCA
        class EEGReshaper(BaseEstimator, TransformerMixin):
            def __init__(self, pca_components=None):
                self.pca_components = pca_components
                self.pca = None
                
            def fit(self, X, y=None):
                if self.pca_components is not None:
                    # Reshape 3D data to 2D for PCA: (epochs, channels, time) -> (epochs, channels*time)
                    X_reshaped = X.reshape(X.shape[0], -1)
                    
                    # Set appropriate number of components based on data size
                    if self.pca_components == 'auto':
                        # Use a fraction of available dimensions, at most n_samples - 1
                        n_samples = X_reshaped.shape[0]
                        n_features = X_reshaped.shape[1]
                        max_components = min(n_samples - 1, n_features)
                        # Use half of max_components but at least 2
                        n_components = max(2, min(max_components // 2, 10))
                        print(f"Auto-selecting {n_components} PCA components based on data size")
                    else:
                        # Ensure n_components doesn't exceed limits
                        n_samples = X_reshaped.shape[0]
                        n_features = X_reshaped.shape[1]
                        max_components = min(n_samples - 1, n_features)
                        n_components = min(self.pca_components, max_components)
                        if n_components != self.pca_components:
                            print(f"Reducing PCA components from {self.pca_components} to {n_components} due to data size limitations")
                    
                    self.pca = PCA(n_components=n_components)
                    self.pca.fit(X_reshaped)
                    self.n_components = n_components
                return self
                
            def transform(self, X):
                if self.pca is not None:
                    # Reshape 3D data to 2D for PCA
                    X_orig_shape = X.shape
                    X_reshaped = X.reshape(X_orig_shape[0], -1)
                    
                    # Apply PCA
                    X_pca = self.pca.transform(X_reshaped)
                    
                    # Reshape back to 3D for CSP: (epochs, pca_components) -> (epochs, pca_components, 1)
                    # CSP will treat each PCA component as a "channel"
                    return X_pca.reshape(X_orig_shape[0], self.n_components, -1)
                else:
                    return X
        
        # Create pipeline components
        pipeline_steps = []
        
        # Add PCA step if requested (using custom reshaper)
        if self.pca_components is not None:
            pipeline_steps.append(('reshaper', EEGReshaper(pca_components=self.pca_components)))
        
        # Add CSP with regularization
        pipeline_steps.append(('csp', CSP(n_components=self.n_components, 
                                         reg='ledoit_wolf',  # Add regularization
                                         log=True, 
                                         norm_trace=False)))
        
        # Add SVM classifier
        pipeline_steps.append(('svm', SVC(kernel=self.kernel, C=self.C, probability=True)))
        
        # Create the pipeline
        self.pipeline = Pipeline(pipeline_steps)
        
        # Define cross-validation strategy
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.pipeline, X, y, cv=cv_strategy, scoring='accuracy')
        
        # Train on all data for final model
        self.pipeline.fit(X, y)
        
        # Return evaluation metrics
        return {
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'cv_scores': cv_scores,
            'model': self.pipeline
        }
    
    def predict(self, X):
        """Make predictions on new data.
        
        Parameters:
        -----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
            The EEG data to predict
        
        Returns:
        --------
        ndarray
            Predicted class labels
        """
        return self.pipeline.predict(X)

class EEGTransformer(nn.Module):
    """1D Transformer for EEG classification."""
    
    def __init__(self, n_channels, n_timepoints, n_classes, d_model=64, nhead=8, 
                 num_encoder_layers=4, dim_feedforward=128, dropout=0.1):
        """Initialize the EEG Transformer model.
        
        Parameters:
        -----------
        n_channels : int
            Number of EEG channels
        n_timepoints : int
            Number of time points in each EEG segment
        n_classes : int
            Number of output classes
        d_model : int
            Dimension of the model
        nhead : int
            Number of heads in multi-head attention
        num_encoder_layers : int
            Number of transformer encoder layers
        dim_feedforward : int
            Dimension of feedforward network
        dropout : float
            Dropout probability
        """
        super(EEGTransformer, self).__init__()
        
        # Channel embedding layer (maps each channel to d_model dimensions)
        self.channel_embedding = nn.Linear(n_timepoints, d_model)
        
        # Positional encoding for channels
        self.pos_encoder = nn.Parameter(torch.zeros(1, n_channels, d_model))
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                  dim_feedforward=dim_feedforward, 
                                                  dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model * n_channels, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes)
        )
    
    def forward(self, x):
        """Forward pass through the model.
        
        Parameters:
        -----------
        x : tensor, shape (batch_size, n_channels, n_timepoints)
            Input EEG data
        
        Returns:
        --------
        tensor, shape (batch_size, n_classes)
            Output class logits
        """
        batch_size, n_channels, n_timepoints = x.shape
        
        # Create channel embeddings
        x = self.channel_embedding(x)  # (batch_size, n_channels, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoder
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)  # (batch_size, n_channels, d_model)
        
        # Flatten and classify
        x = x.reshape(batch_size, -1)  # (batch_size, n_channels * d_model)
        x = self.classifier(x)  # (batch_size, n_classes)
        
        return x

class TransformerClassifier:
    """Transformer-based classifier for EEG data."""
    
    def __init__(self, n_classes, d_model=64, nhead=8, num_encoder_layers=4, 
                 dim_feedforward=128, dropout=0.1, lr=0.001, batch_size=32, 
                 n_epochs=100, weight_decay=0, device=None):
        """Initialize the Transformer classifier.
        
        Parameters:
        -----------
        n_classes : int
            Number of output classes
        d_model : int
            Dimension of the model
        nhead : int
            Number of heads in multi-head attention
        num_encoder_layers : int
            Number of transformer encoder layers
        dim_feedforward : int
            Dimension of feedforward network
        dropout : float
            Dropout probability
        lr : float
            Learning rate
        batch_size : int
            Batch size for training
        n_epochs : int
            Number of training epochs
        weight_decay : float
            L2 regularization strength
        device : str or torch.device
            Device to use for training ('cpu' or 'cuda')
        """
        self.n_classes = n_classes
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.weight_decay = weight_decay
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Model will be initialized in train_and_evaluate
        self.model = None
        self.scaler = StandardScaler()
    
    def _prepare_data(self, X, y, val_split=0.2):
        """Prepare data for training the transformer.
        
        Parameters:
        -----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
            EEG data
        y : ndarray, shape (n_epochs,)
            Class labels
        val_split : float
            Validation split ratio
        
        Returns:
        --------
        tuple
            Training and validation data loaders
        """
        # Scale the data across time for each channel and epoch
        X_reshaped = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
        X_scaled = self.scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(X.shape)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.LongTensor(y)
        
        # Create dataset
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Split into training and validation sets
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train_and_evaluate(self, X, y, val_split=0.2, early_stopping_patience=10):
        """Train and evaluate the transformer model.
        
        Parameters:
        -----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
            EEG data
        y : ndarray, shape (n_epochs,)
            Class labels
        val_split : float
            Validation split ratio
        early_stopping_patience : int
            Number of epochs to wait for improvement before stopping
        
        Returns:
        --------
        dict
            Dictionary containing training history and evaluation metrics
        """
        n_channels, n_timepoints = X.shape[1], X.shape[2]
        
        # Initialize model
        self.model = EEGTransformer(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            n_classes=self.n_classes,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout
        ).to(self.device)
        
        # Print model summary
        print(f"Model architecture: {self.model.__class__.__name__}")
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        # Prepare data
        train_loader, val_loader = self._prepare_data(X, y, val_split)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.lr,
            weight_decay=self.weight_decay  # L2 regularization
        )
        # Add learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
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
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Add some noise to inputs for regularization
                if self.dropout > 0:
                    noise = torch.randn_like(inputs) * 0.05
                    inputs = inputs + noise
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    outputs = self.model(inputs)
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
        
        # Evaluate on the entire dataset
        self.model.eval()
        X_tensor = torch.FloatTensor(self.scaler.transform(X.reshape(X.shape[0] * X.shape[1], X.shape[2])).reshape(X.shape)).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        
        # Calculate final metrics
        final_acc = (predicted.cpu().numpy() == y).mean()
        
        return {
            'history': history,
            'final_accuracy': final_acc,
            'best_val_loss': best_val_loss,
            'model': self.model
        }
    
    def predict(self, X):
        """Make predictions on new data.
        
        Parameters:
        -----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
            EEG data to predict
        
        Returns:
        --------
        ndarray
            Predicted class labels
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Scale the data
        X_scaled = self.scaler.transform(X.reshape(X.shape[0] * X.shape[1], X.shape[2])).reshape(X.shape)
        
        # Convert to tensor and move to device
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        
        return predicted.cpu().numpy()

class EnhancedEEGNet(nn.Module):
    """Enhanced EEGNet model for EEG classification with spectral features and subject adaptation.
    
    Based on: EEGNet (Lawhern et al., 2018) with improvements
    """
    
    def __init__(self, n_channels, n_times, n_classes, 
                 dropout_rate=0.6, kernel_length=32, F1=16, 
                 D=4, F2=32, use_spectral=True, use_adaptation=True):
        super(EnhancedEEGNet, self).__init__()
        
        # Subject adaptation layer (optional)
        self.use_adaptation = use_adaptation
        if use_adaptation:
            self.adaptation = SubjectAdaptationLayer(n_channels)
            
        # Whether to use spectral features
        self.use_spectral = use_spectral
        if use_spectral:
            # For spectral branch
            self.spectral_conv = nn.Sequential(
                nn.Conv1d(n_channels, F2, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(F2),
                nn.ELU(),
                nn.Dropout(dropout_rate)
            )
        
        # Raw EEG branch - Temporal Convolution
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding='same', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)
        
        # Spatial Convolution
        self.depthwise_conv = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Separable Convolution - proper implementation with depthwise followed by pointwise
        # Depthwise part (groups=channels, in_channels=channels, out_channels=channels)
        self.separable_conv_depth = nn.Conv2d(F1 * D, F1 * D, (1, 16), 
                                              padding='same', groups=F1 * D, bias=False)
        # Pointwise part (regular 1x1 convolution to change channel dimensions)
        self.separable_conv_point = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Calculate final output sizes after pooling
        time_after_pooling = n_times // 32
        if time_after_pooling < 1:
            time_after_pooling = 1
        
        # Store dimensions for debugging
        self.n_channels = n_channels
        self.n_times = n_times
        self.time_after_pooling = time_after_pooling
        
        # Final classifier
        # Add spectral features if used
        features_size = F2 * time_after_pooling
        if use_spectral:
            features_size += F2  # Add spectral features size
            
        # Output layer with correct feature size
        self.fc1 = nn.Linear(features_size, F2)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(F2, n_classes)
        
    def forward(self, x, x_spectral=None):
        batch_size = x.size(0)
        
        # Apply subject adaptation if enabled
        if self.use_adaptation:
            x = self.adaptation(x)
        
        # Process raw EEG data
        # x shape: (batch, channels, time)
        x_raw = x.unsqueeze(1)  # (batch, 1, channels, time)
        
        # Block 1
        x_raw = self.conv1(x_raw)
        x_raw = self.batchnorm1(x_raw)
        
        # Block 2: Depthwise Convolution
        x_raw = self.depthwise_conv(x_raw)
        x_raw = self.batchnorm2(x_raw)
        x_raw = F.elu(x_raw)
        x_raw = F.avg_pool2d(x_raw, (1, 4))
        x_raw = self.dropout1(x_raw)
        
        # Block 3: Separable Convolution (depth + point)
        x_raw = self.separable_conv_depth(x_raw)
        x_raw = self.separable_conv_point(x_raw)
        x_raw = self.batchnorm3(x_raw)
        x_raw = F.elu(x_raw)
        x_raw = F.avg_pool2d(x_raw, (1, 8))
        x_raw = self.dropout2(x_raw)
        
        # Flatten
        x_raw = x_raw.reshape(batch_size, -1)
        
        # Process spectral features if provided and enabled
        if self.use_spectral and x_spectral is not None:
            # x_spectral shape: (batch, channels, bands)
            # Transpose to (batch, channels, bands)
            x_spectral = x_spectral.permute(0, 1, 2)
            x_spectral = self.spectral_conv(x_spectral)
            x_spectral = x_spectral.mean(dim=2)  # Global average pooling
            
            # Concatenate raw and spectral features
            x_combined = torch.cat([x_raw, x_spectral], dim=1)
        else:
            x_combined = x_raw
        
        # Check shapes and dynamically adjust if needed
        if x_combined.size(1) != self.fc1.in_features:
            print(f"Shape mismatch: x_combined size is {x_combined.size(1)}, expected {self.fc1.in_features}")
            device = x_combined.device
            self.fc1 = nn.Linear(x_combined.size(1), self.fc1.out_features).to(device)
        
        # Final classification
        x = F.elu(self.fc1(x_combined))
        x = self.dropout3(x)
        x = self.fc2(x)
        
        return x

class EnhancedEEGNetClassifier:
    """Enhanced EEGNet-based classifier with spectral features and advanced training strategies."""
    
    def __init__(self, n_classes, dropout_rate=0.6, kernel_length=32, 
                 F1=16, D=4, F2=32, lr=0.0005, batch_size=64, 
                 n_epochs=300, weight_decay=0.02, device=None,
                 augmentation_level=0.7, use_spectral=True, 
                 use_adaptation=True, label_smoothing=0.1):
        """Initialize the Enhanced EEGNet classifier.
        
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
            Depth multiplier
        F2 : int
            Number of separable filters
        lr : float
            Learning rate
        batch_size : int
            Batch size for training
        n_epochs : int
            Number of training epochs
        weight_decay : float
            L2 regularization strength
        device : str or torch.device
            Device to use for training ('cpu' or 'cuda')
        augmentation_level : float
            Level of data augmentation to apply (0.0-1.0)
        use_spectral : bool
            Whether to use spectral features
        use_adaptation : bool
            Whether to use subject adaptation layer
        label_smoothing : float
            Amount of label smoothing to apply (0.0-1.0)
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
        self.augmentation_level = augmentation_level
        self.use_spectral = use_spectral
        self.use_adaptation = use_adaptation
        self.label_smoothing = label_smoothing
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Model will be initialized in train_and_evaluate
        self.model = None
        self.scaler = StandardScaler()
        self.scaler_spectral = StandardScaler()
    
    def _augment_data(self, X):
        """Apply enhanced data augmentation to EEG data.
        
        Parameters:
        -----------
        X : torch.Tensor
            Input EEG data tensor (batch, channels, time)
            
        Returns:
        --------
        torch.Tensor
            Augmented EEG data
        """
        batch_size, n_channels, n_times = X.shape
        X_aug = X.clone()
        
        # Only augment with probability determined by augmentation_level
        if torch.rand(1).item() < self.augmentation_level:
            # 1. Add random noise with varying intensity
            noise_level = 0.05 * torch.rand(1).item() * self.augmentation_level
            X_aug = X_aug + torch.randn_like(X_aug) * noise_level
            
            # 2. Random time shift (up to 15% of signal length)
            max_shift = int(n_times * 0.15)
            if max_shift > 0:
                shift = torch.randint(-max_shift, max_shift+1, (1,)).item()
                if shift > 0:
                    X_aug[:, :, shift:] = X_aug[:, :, :-shift]
                    X_aug[:, :, :shift] = 0
                elif shift < 0:
                    X_aug[:, :, :shift] = X_aug[:, :, -shift:]
                    X_aug[:, :, shift:] = 0
            
            # 3. Enhanced channel masking (10-30% of channels)
            if torch.rand(1).item() < 0.7:  # Apply with high probability
                mask_prob = 0.1 + 0.2 * torch.rand(1).item()  # Between 10-30%
                mask_channels = torch.rand(batch_size, n_channels, 1) < mask_prob
                X_aug[mask_channels.expand_as(X_aug)] = 0
            
            # 4. Frequency selective dropout (simulate electrode noise)
            if torch.rand(1).item() < 0.3:  # Apply with 30% probability
                # Apply bandstop filter on random channels (Fourier method)
                for i in range(batch_size):
                    if torch.rand(1).item() < 0.5:
                        ch = torch.randint(0, n_channels, (1,)).item()
                        # Apply using frequency domain filter 
                        x_fft = torch.fft.rfft(X_aug[i, ch])
                        # Create notch filter by zeroing out random frequencies
                        mask = torch.ones_like(x_fft)
                        low_bin = torch.randint(0, len(mask)//3, (1,)).item()
                        high_bin = low_bin + torch.randint(1, 10, (1,)).item()
                        mask[low_bin:high_bin] = 0
                        x_fft = x_fft * mask
                        X_aug[i, ch] = torch.fft.irfft(x_fft, n=n_times)
            
            # 5. Time warping (very slight slowing/speeding)
            if torch.rand(1).item() < 0.2:  # Apply with 20% probability
                for i in range(batch_size):
                    if torch.rand(1).item() < 0.3:
                        # Interpolate to slightly stretch/compress the signal
                        warp_factor = 0.9 + 0.2 * torch.rand(1).item()  # 0.9-1.1
                        orig_indices = torch.arange(0, n_times).float()
                        new_indices = torch.linspace(0, n_times-1, int(n_times * warp_factor))
                        if len(new_indices) > n_times:
                            new_indices = new_indices[:n_times]
                        elif len(new_indices) < n_times:
                            # Pad with zeros if needed
                            padded = torch.zeros(n_times)
                            padded[:len(new_indices)] = new_indices
                            new_indices = padded
                            
                        for ch in range(n_channels):
                            signal = X_aug[i, ch]
                            warped = torch.nn.functional.interpolate(
                                signal.unsqueeze(0).unsqueeze(0), 
                                size=len(new_indices), 
                                mode='linear', 
                                align_corners=False
                            ).squeeze()
                            # Ensure it's the right size (handle edge cases)
                            if len(warped) < n_times:
                                padded = torch.zeros(n_times)
                                padded[:len(warped)] = warped
                                warped = padded
                            elif len(warped) > n_times:
                                warped = warped[:n_times]
                            X_aug[i, ch] = warped
        
        return X_aug
    
    def _prepare_data(self, X, y, X_spectral=None, val_split=0.2):
        """Prepare data for training the enhanced EEGNet.
        
        Parameters:
        -----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
            EEG data
        y : ndarray, shape (n_epochs,)
            Class labels
        X_spectral : ndarray or None, shape (n_epochs, n_channels, n_bands)
            Spectral features (if None, not used)
        val_split : float
            Validation split ratio
        
        Returns:
        --------
        tuple
            Training and validation data loaders
        """
        # Scale the raw data
        X_reshaped = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
        X_scaled = self.scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(X.shape)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.LongTensor(y)
        
        # Prepare spectral features if available
        X_spectral_tensor = None
        if self.use_spectral and X_spectral is not None:
            # Scale spectral features (log transform first due to power values)
            X_spectral = np.log(np.abs(X_spectral) + 1e-6)  # Avoid log(0)
            X_spectral_reshaped = X_spectral.reshape(X_spectral.shape[0] * X_spectral.shape[1], X_spectral.shape[2])
            X_spectral_scaled = self.scaler_spectral.fit_transform(X_spectral_reshaped)
            X_spectral_scaled = X_spectral_scaled.reshape(X_spectral.shape)
            X_spectral_tensor = torch.FloatTensor(X_spectral_scaled).to(self.device)
            
            # Create dataset with both raw and spectral features
            dataset = TensorDataset(X_tensor, X_spectral_tensor, y_tensor)
        else:
            # Create dataset with only raw features
            dataset = TensorDataset(X_tensor, y_tensor)
        
        # Split into training and validation sets
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train_and_evaluate(self, X, y, X_spectral=None, val_split=0.2, early_stopping_patience=40):
        """Train and evaluate the Enhanced EEGNet model.
        
        Parameters:
        -----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
            EEG data
        y : ndarray, shape (n_epochs,)
            Class labels
        X_spectral : ndarray or None, shape (n_epochs, n_channels, n_bands)
            Spectral features (if None, will be extracted if use_spectral=True)
        val_split : float
            Validation split ratio
        early_stopping_patience : int
            Number of epochs to wait for improvement before stopping
        
        Returns:
        --------
        dict
            Dictionary containing training history and evaluation metrics
        """
        n_channels, n_times = X.shape[1], X.shape[2]
        
        # Extract spectral features if requested and not provided
        if self.use_spectral and X_spectral is None:
            print("Extracting spectral features...")
            X_spectral = extract_spectral_features(X)
        
        # Initialize model
        self.model = EnhancedEEGNet(
            n_channels=n_channels,
            n_times=n_times,
            n_classes=self.n_classes,
            dropout_rate=self.dropout_rate,
            kernel_length=min(self.kernel_length, n_times // 4),  # Ensure kernel fits
            F1=self.F1,
            D=self.D,
            F2=self.F2,
            use_spectral=self.use_spectral and X_spectral is not None,
            use_adaptation=self.use_adaptation
        ).to(self.device)
        
        # Print model summary
        print(f"Model architecture: {self.model.__class__.__name__}")
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        # Prepare data
        train_loader, val_loader = self._prepare_data(X, y, X_spectral, val_split)
        
        # Loss function with label smoothing
        criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
        # Optimizer with different weight decays for different parts of the model
        # Less regularization for adaptation layer, more for the rest
        if self.use_adaptation:
            adaptation_params = list(self.model.adaptation.parameters())
            other_params = [p for p in self.model.parameters() if p not in set(adaptation_params)]
            
            optimizer = optim.AdamW([
                {'params': adaptation_params, 'weight_decay': self.weight_decay * 0.1},
                {'params': other_params, 'weight_decay': self.weight_decay}
            ], lr=self.lr)
        else:
            optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        # Add learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=15, verbose=True
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
            
            for batch in train_loader:
                # Different processing based on whether spectral features are included
                if len(batch) == 3:  # With spectral features
                    inputs, inputs_spectral, labels = batch
                    inputs, inputs_spectral, labels = inputs.to(self.device), inputs_spectral.to(self.device), labels.to(self.device)
                    
                    # Apply data augmentation during training
                    inputs = self._augment_data(inputs)
                    
                    optimizer.zero_grad()
                    outputs = self.model(inputs, inputs_spectral)
                else:  # Without spectral features
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    # Apply data augmentation during training
                    inputs = self._augment_data(inputs)
                    
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    # Different processing based on whether spectral features are included
                    if len(batch) == 3:  # With spectral features
                        inputs, inputs_spectral, labels = batch
                        inputs, inputs_spectral, labels = inputs.to(self.device), inputs_spectral.to(self.device), labels.to(self.device)
                        outputs = self.model(inputs, inputs_spectral)
                    else:  # Without spectral features
                        inputs, labels = batch
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self.model(inputs)
                    
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
        
        # Evaluate on the entire dataset
        self.model.eval()
        X_tensor = torch.FloatTensor(self.scaler.transform(X.reshape(X.shape[0] * X.shape[1], X.shape[2])).reshape(X.shape)).to(self.device)
        
        # Prepare spectral features for evaluation if available
        X_spectral_tensor = None
        if self.use_spectral and X_spectral is not None:
            # Log transform and scale spectral features
            X_spectral = np.log(np.abs(X_spectral) + 1e-6)
            X_spectral_reshaped = X_spectral.reshape(X_spectral.shape[0] * X_spectral.shape[1], X_spectral.shape[2])
            X_spectral_scaled = self.scaler_spectral.transform(X_spectral_reshaped).reshape(X_spectral.shape)
            X_spectral_tensor = torch.FloatTensor(X_spectral_scaled).to(self.device)
        
        with torch.no_grad():
            if X_spectral_tensor is not None:
                outputs = self.model(X_tensor, X_spectral_tensor)
            else:
                outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        
        # Calculate final metrics
        final_acc = (predicted.cpu().numpy() == y).mean()
        
        return {
            'history': history,
            'final_accuracy': final_acc,
            'best_val_loss': best_val_loss,
            'model': self.model
        }
    
    def predict(self, X, X_spectral=None):
        """Make predictions on new data.
        
        Parameters:
        -----------
        X : ndarray, shape (n_epochs, n_channels, n_times)
            EEG data to predict
        X_spectral : ndarray or None, shape (n_epochs, n_channels, n_bands)
            Spectral features (if None, will be extracted if use_spectral=True)
        
        Returns:
        --------
        ndarray
            Predicted class labels
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Extract spectral features if requested and not provided
        if self.use_spectral and X_spectral is None and hasattr(self, 'scaler_spectral'):
            X_spectral = extract_spectral_features(X)
        
        # Scale the raw data
        X_scaled = self.scaler.transform(X.reshape(X.shape[0] * X.shape[1], X.shape[2])).reshape(X.shape)
        
        # Convert to tensor and move to device
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Prepare spectral features if available
        X_spectral_tensor = None
        if self.use_spectral and X_spectral is not None and hasattr(self, 'scaler_spectral'):
            # Log transform and scale spectral features
            X_spectral = np.log(np.abs(X_spectral) + 1e-6)
            X_spectral_scaled = self.scaler_spectral.transform(X_spectral.reshape(X_spectral.shape[0] * X_spectral.shape[1], X_spectral.shape[2])).reshape(X_spectral.shape)
            X_spectral_tensor = torch.FloatTensor(X_spectral_scaled).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            if X_spectral_tensor is not None:
                outputs = self.model(X_tensor, X_spectral_tensor)
            else:
                outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        
        return predicted.cpu().numpy()

def run_classification(subject_id, run_ids, strategy='csp_svm', tmin=0.0, tmax=2.0, 
                      event_ids=None, multi_subject=False, use_motor_channels=True, **kwargs):
    """Run EEG classification using the specified strategy.
    
    Parameters:
    -----------
    subject_id : str or list of str
        Subject ID or list of subject IDs
    run_ids : list of str
        List of run IDs to use
    strategy : str
        Classification strategy ('csp_svm', 'transformer', 'eegnet', or 'enhanced_eegnet')
    tmin, tmax : float
        Start and end time for epochs to use for classification
    event_ids : dict or None
        Dictionary mapping event names to IDs. If None, use all events.
    multi_subject : bool
        Whether to load data from multiple subjects
    use_motor_channels : bool
        Whether to select only motor-related channels
    **kwargs : dict
        Additional parameters for the specific classifier
    
    Returns:
    --------
    dict
        Results of the classification
    """
    # Load data from all runs/subjects
    if multi_subject:
        epochs = load_multi_subject_data(subject_id, run_ids)
    else:
        epochs = load_processed_data(subject_id, run_ids)
    
    # Select only motor-related channels if requested
    if use_motor_channels:
        original_channels = len(epochs.ch_names)
        epochs = select_motor_channels(epochs)
        print(f"Selected {len(epochs.ch_names)} motor-related channels out of {original_channels} total channels")
    
    # Adjust tmax if needed
    if tmax > epochs.tmax:
        print(f"Warning: tmax={tmax} exceeds available time range. Setting to {epochs.tmax}")
        tmax = epochs.tmax
    
    # Apply preprocessing based on strategy
    if strategy in ['enhanced_eegnet']:
        # Apply enhanced preprocessing including filtering
        preprocessing_params = kwargs.pop('preprocessing_params', {})
        fmin = preprocessing_params.get('fmin', 8)
        fmax = preprocessing_params.get('fmax', 30)
        preprocessed_epochs = enhanced_preprocessing(epochs, fmin=fmin, fmax=fmax)
    else:
        preprocessed_epochs = epochs
    
    # Extract features and labels
    X, y = extract_features_and_labels(preprocessed_epochs, event_ids, tmin, tmax)
    
    # Extract spectral features if using enhanced eegnet
    X_spectral = None
    if strategy == 'enhanced_eegnet':
        # Use sampling frequency from the epochs
        sfreq = preprocessed_epochs.info['sfreq']
        X_spectral = extract_spectral_features(X, sfreq=sfreq)
    
    # Run classification based on the specified strategy
    if strategy == 'csp_svm':
        # Default parameters
        params = {
            'n_components': 4,
            'kernel': 'linear',
            'C': 1.0,
            'cv': 5,
            'pca_components': None  # Default to no PCA when using motor channels
        }
        # Update with user-provided parameters
        params.update(kwargs)
        
        # Create and train the classifier
        classifier = CSPSVMClassifier(
            n_components=params['n_components'],
            kernel=params['kernel'],
            C=params['C'],
            pca_components=params['pca_components']
        )
        results = classifier.train_and_evaluate(X, y, cv=params['cv'])
        
    elif strategy == 'transformer':
        # Default parameters
        params = {
            'd_model': 16,
            'nhead': 2,
            'num_encoder_layers': 1,
            'dim_feedforward': 32,
            'dropout': 0.5,
            'lr': 0.0001,
            'batch_size': 32,
            'n_epochs': 200,
            'weight_decay': 0.01,
            'early_stopping_patience': 20
        }
        # Update with user-provided parameters
        params.update(kwargs)
        
        # Create and train the classifier
        classifier = TransformerClassifier(
            n_classes=len(np.unique(y)),
            d_model=params['d_model'],
            nhead=params['nhead'],
            num_encoder_layers=params['num_encoder_layers'],
            dim_feedforward=params['dim_feedforward'],
            dropout=params['dropout'],
            lr=params['lr'],
            batch_size=params['batch_size'],
            n_epochs=params['n_epochs'],
            weight_decay=params['weight_decay']
        )
        results = classifier.train_and_evaluate(X, y, early_stopping_patience=params['early_stopping_patience'])
    
    elif strategy == 'eegnet':
        # Default parameters for EEGNet
        params = {
            'dropout_rate': 0.5,
            'kernel_length': 64,
            'F1': 8,
            'D': 2,
            'F2': 16,
            'lr': 0.001,
            'batch_size': 32,
            'n_epochs': 200,
            'weight_decay': 0.01,
            'early_stopping_patience': 20,
            'augmentation_level': 0.3
        }
        # Update with user-provided parameters
        params.update(kwargs)
        
        # Create and train the classifier
        classifier = EnhancedEEGNetClassifier(
            n_classes=len(np.unique(y)),
            dropout_rate=params['dropout_rate'],
            kernel_length=params['kernel_length'],
            F1=params['F1'],
            D=params['D'],
            F2=params['F2'],
            lr=params['lr'],
            batch_size=params['batch_size'],
            n_epochs=params['n_epochs'],
            weight_decay=params['weight_decay'],
            augmentation_level=params['augmentation_level']
        )
        results = classifier.train_and_evaluate(X, y, early_stopping_patience=params['early_stopping_patience'])
    
    elif strategy == 'enhanced_eegnet':
        # Default parameters for Enhanced EEGNet
        params = {
            'dropout_rate': 0.6,
            'kernel_length': 32,
            'F1': 16,
            'D': 4,
            'F2': 32,
            'lr': 0.0005,
            'batch_size': 64,
            'n_epochs': 300,
            'weight_decay': 0.02,
            'early_stopping_patience': 40,
            'augmentation_level': 0.7,
            'use_spectral': True,
            'use_adaptation': True,
            'label_smoothing': 0.1
        }
        # Update with user-provided parameters
        params.update(kwargs)
        
        # Create and train the classifier
        classifier = EnhancedEEGNetClassifier(
            n_classes=len(np.unique(y)),
            dropout_rate=params['dropout_rate'],
            kernel_length=params['kernel_length'],
            F1=params['F1'],
            D=params['D'],
            F2=params['F2'],
            lr=params['lr'],
            batch_size=params['batch_size'],
            n_epochs=params['n_epochs'],
            weight_decay=params['weight_decay'],
            augmentation_level=params['augmentation_level'],
            use_spectral=params['use_spectral'],
            use_adaptation=params['use_adaptation'],
            label_smoothing=params['label_smoothing']
        )
        results = classifier.train_and_evaluate(X, y, X_spectral=X_spectral, 
                                              early_stopping_patience=params['early_stopping_patience'])
        
    else:
        raise ValueError(f"Unknown classification strategy: {strategy}")
    
    # Add classifier and data information to results
    results['classifier'] = classifier
    results['X'] = X
    results['y'] = y
    
    # Store channel information for visualization
    results['info'] = epochs.info
    
    return results

def run_enhanced_eegnet_classification(subject_ids, run_ids, multi_subject=True, 
                                    fmin=8, fmax=30, ensemble_size=1, use_motor_channels=True):
    """Run enhanced EEGNet classification with optimal settings.
    
    Parameters:
    -----------
    subject_ids : list or str
        List of subject IDs to use
    run_ids : list
        List of run IDs to use
    multi_subject : bool
        Whether to load data from multiple subjects
    fmin, fmax : float
        Frequency band to filter data
    ensemble_size : int
        Number of models to train for ensemble (>1 creates an ensemble)
    use_motor_channels : bool
        Whether to select only motor-related channels
    
    Returns:
    --------
    dict
        Classification results
    """
    # Time window optimized for motor imagery
    best_time_window = (0.5, 3.0)  
    t_min, t_max = best_time_window
    
    print("\nRunning Enhanced EEGNet classification...")
    if use_motor_channels:
        print("Using only motor-related channels")
    
    # If ensemble_size > 1, train multiple models and average predictions
    if ensemble_size > 1:
        print(f"Training ensemble of {ensemble_size} models")
        
        # We'll store all models here
        ensemble_models = []
        all_histories = []
        val_accuracies = []
        
        # Train models with different random seeds
        for i in range(ensemble_size):
            print(f"\n--- Training model {i+1}/{ensemble_size} ---")
            
            # Set different random seed for each model
            torch.manual_seed(42 + i)
            np.random.seed(42 + i)
            
            # Run classification with enhanced EEGNet parameters and track the model
            result = run_classification(
                subject_ids, run_ids, strategy='enhanced_eegnet',
                tmin=t_min, tmax=t_max,
                preprocessing_params={'fmin': fmin, 'fmax': fmax},
                # Slightly vary hyperparameters for diversity
                dropout_rate=0.5 + 0.1 * np.random.random(),
                kernel_length=int(30 + 6 * np.random.random()),
                F1=int(14 + 4 * np.random.random()),
                D=int(3 + 2 * np.random.random()),
                F2=int(28 + 8 * np.random.random()),
                lr=0.0005 * (0.8 + 0.4 * np.random.random()),
                weight_decay=0.02 * (0.8 + 0.4 * np.random.random()),
                multi_subject=multi_subject,
                use_motor_channels=use_motor_channels
            )
            
            ensemble_models.append(result['classifier'])
            all_histories.append(result['history'])
            val_accuracies.append(result['final_accuracy'])
            
            print(f"Model {i+1} accuracy: {result['final_accuracy']:.4f}")
        
        # Combine results
        overall_result = {
            'ensemble_models': ensemble_models,
            'all_histories': all_histories,
            'val_accuracies': val_accuracies,
            'final_accuracy': np.mean(val_accuracies),
            'std_accuracy': np.std(val_accuracies) 
        }
        
        print(f"\nEnsemble accuracy: {overall_result['final_accuracy']:.4f} ± {overall_result['std_accuracy']:.4f}")
        return overall_result
        
    else:
        # Run single model classification with enhanced EEGNet parameters
        enhanced_results = run_classification(
            subject_ids, run_ids, strategy='enhanced_eegnet',
            tmin=t_min, tmax=t_max,
            preprocessing_params={'fmin': fmin, 'fmax': fmax},
            # Enhanced EEGNet parameters
            dropout_rate=0.6,
            kernel_length=32,
            F1=16,
            D=4,
            F2=32,
            lr=0.0005,
            batch_size=64,
            n_epochs=300,
            weight_decay=0.02,
            early_stopping_patience=40,
            augmentation_level=0.7,
            use_spectral=True,
            use_adaptation=True,
            label_smoothing=0.1,
            multi_subject=multi_subject,
            use_motor_channels=use_motor_channels
        )
        
        # Plot results
        plot_results(enhanced_results, 'enhanced_eegnet')
        
        return enhanced_results

def plot_results(results, strategy):
    """Plot the results of the classification.
    
    Parameters:
    -----------
    results : dict
        Results from run_classification
    strategy : str
        Classification strategy used
    """
    if strategy == 'csp_svm':
        # Plot cross-validation scores
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(results['cv_scores'])), results['cv_scores'])
        plt.axhline(y=results['cv_accuracy_mean'], color='r', linestyle='-', 
                   label=f'Mean: {results["cv_accuracy_mean"]:.3f}')
        plt.xlabel('Fold')
        plt.ylabel('Accuracy')
        plt.title('Cross-Validation Accuracy Scores (CSP + SVM)')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Plot CSP patterns if available
        csp = results['model'].named_steps['csp']
        if hasattr(csp, 'patterns_') and 'info' in results:
            n_components = csp.n_components
            fig, axes = plt.subplots(1, n_components, figsize=(4 * n_components, 4))
            
            if n_components == 1:
                axes = [axes]
                
            for i, ax in enumerate(axes):
                mne.viz.plot_topomap(csp.patterns_[:, i], 
                                    results['info'],  # Use the stored info
                                    axes=ax, show=False)
                ax.set_title(f'CSP Pattern {i+1}')
            
            plt.tight_layout()
            plt.show()
        else:
            print("CSP patterns or channel information not available for plotting")
            
    elif strategy in ['transformer', 'eegnet']:
        # Plot training history
        history = results['history']
        epochs = range(1, len(history['train_loss']) + 1)
        
        plt.figure(figsize=(12, 5))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
        plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        plt.title(f'Training and Validation Loss ({strategy.capitalize()})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
        plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
        plt.title(f'Training and Validation Accuracy ({strategy.capitalize()})')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        print(f"Final accuracy: {results['final_accuracy']:.4f}")

def hyperparameter_sweep(subject_id, run_ids, tmin=0.0, tmax=2.0, event_ids=None, multi_subject=False):
    """Perform hyperparameter sweep for CSP+SVM model.
    
    Parameters:
    -----------
    subject_id : str or list of str
        Subject ID or list of subject IDs
    run_ids : list of str
        List of run IDs to use
    tmin, tmax : float
        Start and end time for epochs to use for classification
    event_ids : dict or None
        Dictionary mapping event names to IDs. If None, use all events.
    multi_subject : bool
        Whether to load data from multiple subjects
    
    Returns:
    --------
    dict
        Results of the best hyperparameter configuration
    """
    # Load data from all runs/subjects
    if multi_subject:
        epochs = load_multi_subject_data(subject_id, run_ids)
    else:
        epochs = load_processed_data(subject_id, run_ids)
    
    # Adjust tmax if needed
    if tmax > epochs.tmax:
        print(f"Warning: tmax={tmax} exceeds available time range. Setting to {epochs.tmax}")
        tmax = epochs.tmax
    
    # Define parameter grid
    param_grid = {
        'n_components': [2, 4, 6, 8],                  # CSP components
        'kernel': ['linear'],                   # SVM kernel
        'C': [0.1, 1.0, 10.0],                         # SVM regularization
        'pca_components': [None, 'auto', 5, 10],       # PCA dimensionality reduction
        'time_windows': [(0.0, 1.0), (0.0, 2.0), (0.5, 1.5), (0.5, 2.0)], # Time windows
        'frequency_bands': [(8, 13), (13, 30), (8, 30)] # Alpha, Beta, and Alpha+Beta
    }
    
    # Store results
    all_results = []
    best_accuracy = 0
    best_params = None
    best_result = None
    
    # Total number of combinations
    total_combinations = (len(param_grid['n_components']) * 
                          len(param_grid['kernel']) * 
                          len(param_grid['C']) * 
                          len(param_grid['pca_components']) *
                          len(param_grid['time_windows']) *
                          len(param_grid['frequency_bands']))
    
    print(f"Starting hyperparameter sweep with {total_combinations} combinations...")
    
    # Current combination counter
    current = 0
    
    # Iterate over all parameter combinations
    for n_components in param_grid['n_components']:
        for kernel in param_grid['kernel']:
            for C in param_grid['C']:
                for pca_components in param_grid['pca_components']:
                    for time_window in param_grid['time_windows']:
                        t_min, t_max = time_window
                        # Adjust if beyond epoch limits
                        if t_max > epochs.tmax:
                            t_max = epochs.tmax
                            
                        for freq_band in param_grid['frequency_bands']:
                            current += 1
                            f_min, f_max = freq_band
                            
                            print(f"Testing combination {current}/{total_combinations}: " +
                                  f"CSP components={n_components}, kernel={kernel}, C={C}, " +
                                  f"PCA={pca_components}, time=({t_min}, {t_max}), freq=({f_min}, {f_max})")
                            
                            try:
                                # Filter data for this frequency band
                                epochs_filtered = epochs.copy().filter(l_freq=f_min, h_freq=f_max, verbose=False)
                                
                                # Extract features and labels
                                X, y = extract_features_and_labels(epochs_filtered, event_ids, t_min, t_max)
                                
                                # Create and train classifier
                                classifier = CSPSVMClassifier(
                                    n_components=n_components,
                                    kernel=kernel,
                                    C=C,
                                    pca_components=pca_components
                                )
                                result = classifier.train_and_evaluate(X, y, cv=5)
                                
                                # Store configuration and result
                                config = {
                                    'n_components': n_components,
                                    'kernel': kernel,
                                    'C': C,
                                    'pca_components': pca_components,
                                    'time_window': time_window,
                                    'frequency_band': freq_band,
                                    'accuracy': result['cv_accuracy_mean'],
                                    'std': result['cv_accuracy_std']
                                }
                                all_results.append(config)
                                
                                # Check if this is the best so far
                                if result['cv_accuracy_mean'] > best_accuracy:
                                    best_accuracy = result['cv_accuracy_mean']
                                    best_params = config
                                    best_result = result
                                    print(f"New best accuracy: {best_accuracy:.4f} with {config}")
                            except Exception as e:
                                print(f"Error with combination: {e}")
                                continue
    
    # Sort results by accuracy
    all_results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    # Print top 5 results
    print("\nTop 5 parameter combinations:")
    for i, result in enumerate(all_results[:5]):
        print(f"{i+1}. Accuracy: {result['accuracy']:.4f} ± {result['std']:.4f}")
        print(f"   CSP components: {result['n_components']}")
        print(f"   Kernel: {result['kernel']}")
        print(f"   C: {result['C']}")
        print(f"   PCA: {result['pca_components']}")
        print(f"   Time window: {result['time_window']}")
        print(f"   Frequency band: {result['frequency_band']}")
        print("")
    
    # Create visualization of parameter impacts
    if len(all_results) > 0:
        visualize_parameter_impacts(all_results)
    
    return {'best_params': best_params, 'best_result': best_result, 'all_results': all_results}

def visualize_parameter_impacts(results):
    """Visualize the impact of different parameters on classification accuracy.
    
    Parameters:
    -----------
    results : list
        List of dictionaries containing parameter configurations and results
    """
    # Prepare data
    param_names = ['n_components', 'kernel', 'C', 'pca_components', 'time_window', 'frequency_band']
    
    # Create figure with subplots for each parameter
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    axes = axes.flatten()
    
    # Plot for CSP components
    components = sorted(set(r['n_components'] for r in results))
    acc_by_comp = {c: [r['accuracy'] for r in results if r['n_components'] == c] for c in components}
    axes[0].boxplot([acc_by_comp[c] for c in components], labels=components)
    axes[0].set_title('Impact of CSP Components')
    axes[0].set_xlabel('Number of Components')
    axes[0].set_ylabel('Accuracy')
    
    # Plot for kernel
    kernels = sorted(set(r['kernel'] for r in results))
    acc_by_kernel = {k: [r['accuracy'] for r in results if r['kernel'] == k] for k in kernels}
    axes[1].boxplot([acc_by_kernel[k] for k in kernels], labels=kernels)
    axes[1].set_title('Impact of SVM Kernel')
    axes[1].set_xlabel('Kernel Type')
    axes[1].set_ylabel('Accuracy')
    
    # Plot for C values
    c_values = sorted(set(r['C'] for r in results))
    acc_by_c = {c: [r['accuracy'] for r in results if r['C'] == c] for c in c_values}
    axes[2].boxplot([acc_by_c[c] for c in c_values], labels=[str(c) for c in c_values])
    axes[2].set_title('Impact of SVM Regularization (C)')
    axes[2].set_xlabel('C Value')
    axes[2].set_ylabel('Accuracy')
    
    # Plot for PCA
    pca_values = sorted(set(str(r['pca_components']) for r in results))
    acc_by_pca = {str(p): [r['accuracy'] for r in results if str(r['pca_components']) == p] for p in pca_values}
    axes[3].boxplot([acc_by_pca[p] for p in pca_values], labels=pca_values)
    axes[3].set_title('Impact of PCA Components')
    axes[3].set_xlabel('PCA Components')
    axes[3].set_ylabel('Accuracy')
    
    # Plot for time windows
    time_windows = sorted(set(str(r['time_window']) for r in results))
    acc_by_time = {str(t): [r['accuracy'] for r in results if str(r['time_window']) == t] for t in time_windows}
    axes[4].boxplot([acc_by_time[t] for t in time_windows], labels=[t.replace("(", "").replace(")", "") for t in time_windows])
    axes[4].set_title('Impact of Time Window')
    axes[4].set_xlabel('Time Window (start, end)')
    axes[4].set_ylabel('Accuracy')
    
    # Plot for frequency bands
    freq_bands = sorted(set(str(r['frequency_band']) for r in results))
    acc_by_freq = {str(f): [r['accuracy'] for r in results if str(r['frequency_band']) == f] for f in freq_bands}
    axes[5].boxplot([acc_by_freq[f] for f in freq_bands], labels=[f.replace("(", "").replace(")", "") for f in freq_bands])
    axes[5].set_title('Impact of Frequency Band')
    axes[5].set_xlabel('Frequency Band (min, max)')
    axes[5].set_ylabel('Accuracy')
    
    plt.tight_layout()
    plt.show()

    # Create heatmap of n_components vs time_window
    try:
        components = sorted(set(r['n_components'] for r in results))
        time_windows = sorted(set(str(r['time_window']) for r in results))
        heatmap_data = np.zeros((len(components), len(time_windows)))
        
        for i, comp in enumerate(components):
            for j, tw in enumerate(time_windows):
                relevant_results = [r['accuracy'] for r in results 
                                  if r['n_components'] == comp and str(r['time_window']) == tw]
                if relevant_results:
                    heatmap_data[i, j] = np.mean(relevant_results)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
        plt.colorbar(label='Accuracy')
        plt.xticks(np.arange(len(time_windows)), [t.replace("(", "").replace(")", "") for t in time_windows], rotation=45)
        plt.yticks(np.arange(len(components)), components)
        plt.xlabel('Time Window')
        plt.ylabel('CSP Components')
        plt.title('Heatmap: CSP Components vs Time Window')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error creating heatmap: {e}")

def load_multi_subject_data(subject_ids, run_ids):
    """Load processed epochs data for multiple subjects and runs.
    
    Parameters:
    -----------
    subject_ids : list of str
        List of subject IDs to load
    run_ids : list of str
        List of run IDs to load for each subject
    
    Returns:
    --------
    mne.Epochs
        Combined epochs from all subjects and runs
    """
    all_subject_epochs = []
    processed_subjects = 0
    sfreq_count = {}     # Count of each sampling frequency
    subject_epochs_dict = {}
    excluded_subjects = []
    
    # First pass: load all epochs and count sampling frequencies
    for subject_id in subject_ids:
        try:
            # Load data for this subject
            epochs = load_processed_data(subject_id, run_ids)
            sfreq = epochs.info['sfreq']
            
            # Count this sampling frequency
            if sfreq not in sfreq_count:
                sfreq_count[sfreq] = 0
            sfreq_count[sfreq] += 1
            
            # Store the epochs
            subject_epochs_dict[subject_id] = epochs
            processed_subjects += 1
            
        except FileNotFoundError as e:
            print(f"Warning: Could not load data for subject {subject_id}: {str(e)}")
            continue
    
    if not subject_epochs_dict:
        raise ValueError("No data could be loaded for any subject")
    
    # Find the most common sampling frequency
    target_sfreq = max(sfreq_count, key=sfreq_count.get)
    print(f"Target sampling frequency: {target_sfreq} Hz (used by {sfreq_count[target_sfreq]} subjects)")
    
    # Second pass: include only subjects with matching frequency
    for subject_id, epochs in subject_epochs_dict.items():
        current_sfreq = epochs.info['sfreq']
        
        if current_sfreq != target_sfreq:
            print(f"Excluding subject {subject_id} with sampling frequency {current_sfreq} Hz (different from target {target_sfreq} Hz)")
            excluded_subjects.append(subject_id)
            continue
        
        all_subject_epochs.append(epochs)
    
    # Check if we have any subjects left
    if not all_subject_epochs:
        raise ValueError("No subjects with matching sampling frequency found")
    
    # Combine epochs from all included subjects
    combined_epochs = mne.concatenate_epochs(all_subject_epochs)
    print(f"Loaded {len(combined_epochs)} total epochs from {processed_subjects - len(excluded_subjects)} subjects " +
          f"(excluded {len(excluded_subjects)} subjects with different sampling rates)")
    return combined_epochs

def run_csp_svm_classification(subject_ids, run_ids, multi_subject=True):
    """Run CSP+SVM classification with best parameters.
    
    Parameters:
    -----------
    subject_ids : list or str
        List of subject IDs to use
    run_ids : list
        List of run IDs to use
    multi_subject : bool
        Whether to load data from multiple subjects
    
    Returns:
    --------
    dict
        Classification results
    """
    # Best parameters based on previous sweep
    best_params = {
        'n_components': 4,
        'kernel': 'linear',
        'C': 10.0,
        'pca_components': 10,
        'time_window': (0.5, 3.0),  # Updated to use more of the new epoch length
        'frequency_band': (8, 13)   # Alpha band
    }
    
    print("\nRunning CSP+SVM with best parameters...")
    f_min, f_max = best_params['frequency_band']
    t_min, t_max = best_params['time_window']
    
    # Load and filter data
    if multi_subject:
        epochs = load_multi_subject_data(subject_ids, run_ids)
    else:
        epochs = load_processed_data(subject_ids, run_ids)
    
    epochs_filtered = epochs.copy().filter(l_freq=f_min, h_freq=f_max, verbose=False)
    
    # Run classification
    csp_results = run_classification(
        subject_ids, run_ids, strategy='csp_svm',
        tmin=t_min, tmax=t_max,
        n_components=best_params['n_components'],
        kernel=best_params['kernel'],
        C=best_params['C'],
        pca_components=best_params['pca_components'],
        multi_subject=multi_subject
    )
    
    # Plot results
    plot_results(csp_results, 'csp_svm')
    
    return csp_results

def run_transformer_classification(subject_ids, run_ids, multi_subject=True):
    """Run improved Transformer classification.
    
    Parameters:
    -----------
    subject_ids : list or str
        List of subject IDs to use
    run_ids : list
        List of run IDs to use
    multi_subject : bool
        Whether to load data from multiple subjects
    
    Returns:
    --------
    dict
        Classification results
    """
    # Use the same time window as CSP but with transformer-specific parameters
    best_time_window = (0.5, 3.0)  # Updated to use more of the new epoch length
    t_min, t_max = best_time_window
    
    print("\nRunning improved Transformer classification with stronger regularization...")
    
    # Run classification with drastically improved transformer parameters
    transformer_results = run_classification(
        subject_ids, run_ids, strategy='transformer',
        tmin=t_min, tmax=t_max,
        # Much smaller model to reduce capacity
        d_model=16,              # Even smaller model (was 32)
        nhead=2,                 # Fewer attention heads (was 4)
        num_encoder_layers=1,    # Just one layer (was 2)
        dim_feedforward=32,      # Smaller feedforward (was 64)
        # Much stronger regularization
        dropout=0.5,             # Very high dropout (was 0.3)
        lr=0.0001,               # Much lower learning rate (was 0.0005)
        # Training protocol changes
        batch_size=32,           # Larger batch size for better gradient estimates
        n_epochs=200,            # More epochs but with early stopping
        # Pass additional parameters for TransformerClassifier
        early_stopping_patience=20,  # Wait longer before stopping
        weight_decay=0.01,       # Add L2 regularization
        multi_subject=multi_subject
    )
    
    # Plot results
    plot_results(transformer_results, 'transformer')
    
    return transformer_results

def run_eegnet_classification(subject_ids, run_ids, multi_subject=True):
    """Run EEGNet classification.
    
    Parameters:
    -----------
    subject_ids : list or str
        List of subject IDs to use
    run_ids : list
        List of run IDs to use
    multi_subject : bool
        Whether to load data from multiple subjects
    
    Returns:
    --------
    dict
        Classification results
    """
    # Time window and frequency band (based on motor imagery literature)
    best_time_window = (0.5, 3.0)  
    t_min, t_max = best_time_window
    
    print("\nRunning EEGNet classification...")
    
    # Run classification with EEGNet parameters
    eegnet_results = run_classification(
        subject_ids, run_ids, strategy='eegnet',
        tmin=t_min, tmax=t_max,
        # EEGNet parameters
        dropout_rate=0.5,
        kernel_length=64,
        F1=8,
        D=2,
        F2=16,
        lr=0.001,
        batch_size=32,
        n_epochs=200,
        weight_decay=0.01,
        early_stopping_patience=20,
        augmentation_level=0.3,
        multi_subject=multi_subject
    )
    
    # Plot results
    plot_results(eegnet_results, 'eegnet')
    
    return eegnet_results

class SubjectAdaptationLayer(nn.Module):
    """Subject adaptation layer to handle differences between subjects in EEG recordings.
    
    This layer learns channel-wise scaling and bias parameters to adapt to subject-specific
    characteristics, making the model more robust to cross-subject differences.
    """
    
    def __init__(self, n_channels, init_scale=1.0, init_bias=0.0):
        """Initialize the subject adaptation layer.
        
        Parameters:
        -----------
        n_channels : int
            Number of EEG channels
        init_scale : float
            Initial value for scaling parameters
        init_bias : float
            Initial value for bias parameters
        """
        super(SubjectAdaptationLayer, self).__init__()
        
        # Learnable scaling parameters for each channel
        self.scale = nn.Parameter(torch.ones(n_channels) * init_scale)
        
        # Learnable bias parameters for each channel
        self.bias = nn.Parameter(torch.zeros(n_channels) + init_bias)
    
    def forward(self, x):
        """Apply channel-wise scaling and bias.
        
        Parameters:
        -----------
        x : tensor, shape (batch_size, n_channels, n_times)
            Input EEG data
        
        Returns:
        --------
        tensor, shape (batch_size, n_channels, n_times)
            Transformed EEG data
        """
        # x has shape (batch_size, n_channels, n_times)
        # scale has shape (n_channels)
        # bias has shape (n_channels)
        
        # Reshape scale and bias for broadcasting
        scale = self.scale.view(1, -1, 1)
        bias = self.bias.view(1, -1, 1)
        
        # Apply channel-wise scaling and bias
        return x * scale + bias

def select_motor_channels(epochs):
    """Select only electrodes relevant for motor imagery.
    
    Parameters:
    -----------
    epochs : mne.Epochs
        The epochs object containing all channels
        
    Returns:
    --------
    mne.Epochs
        Epochs with only motor-related channels
    """
    # Define motor-related channels (adjust based on your electrode montage)
    motor_channels = ['C3', 'C4', 'Cz', 'FC3', 'FC4', 'FCz', 
                     'CP3', 'CP4', 'CPz', 'C1', 'C2', 'C5', 'C6',
                     # Include some parietal electrodes that may be relevant
                     'P3', 'P4', 'Pz']
    
    # Convert to uppercase to make case-insensitive matching
    available_channels = []
    for ch in motor_channels:
        # Check for exact match and for channel names with index suffixes (e.g., C3_1)
        matches = [name for name in epochs.ch_names if 
                  (name.upper() == ch.upper() or 
                   name.upper().startswith(ch.upper() + '_'))]
        available_channels.extend(matches)
    
    if len(available_channels) < 3:
        print("Warning: Few motor channels found. Using all channels instead.")
        return epochs
        
    print(f"Selecting {len(available_channels)} motor-related channels: {available_channels}")
    
    # Pick only these channels
    return epochs.pick_channels(available_channels)

if __name__ == "__main__":
    # Multi-subject classification
    subject_ids = [f"{i:03d}" for i in range(1, 110)]  # 001 through 109
    run_ids = ['3', '7', '11']  # Motor imagery runs
    
    print("\nLoading data from multiple subjects...")
    try:
        # Run different classification strategies
        # Uncomment the one you want to use
        
        # CSP+SVM classification
        # csp_results = run_csp_svm_classification(subject_ids, run_ids)
        
        # Transformer classification
        # transformer_results = run_transformer_classification(subject_ids, run_ids)
        
        # Basic EEGNet classification
        # eegnet_results = run_eegnet_classification(subject_ids, run_ids)
        
        # Enhanced EEGNet with all improvements
        enhanced_results = run_enhanced_eegnet_classification(
            subject_ids, run_ids, 
            multi_subject=True,
            fmin=8, fmax=30,  # mu + beta rhythm
            ensemble_size=1,  # Use single model (change to >1 for ensemble)
            use_motor_channels=True  # Use only motor-related channels
        )
        
    except Exception as e:
        print(f"Error in multi-subject processing: {str(e)}")
        import traceback
        traceback.print_exc()