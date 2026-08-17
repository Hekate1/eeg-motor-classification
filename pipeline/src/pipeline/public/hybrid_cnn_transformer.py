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
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
import warnings
from copy import deepcopy
from tqdm import tqdm
import copy

warnings.filterwarnings('ignore')

from hycnn_blocks import HybridCNNTransformer
from hycnn_mamba_blocks import HybridCNNMamba

from data_utils import FeatureNormalizer
from feature_modules import CSPModule, SpectrogramModule

# Global debug flag
DEBUG = False
RANDOM_SEED = 42

# Set device for PyTorch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cache for loaded models and their configurations to avoid repeated disk I/O
_load_model_cache = {}

# Debug print function
def dprint(*args, **kwargs):
    """Only print if DEBUG is True"""
    if DEBUG:
        print(*args, **kwargs)

class HybridModelClassifier:
    """
    Wrapper class for training and evaluating the Hybrid CNN/Transformer model
    """
    def __init__(self, n_classes, embedding_dim=128, n_heads=4, n_layers=2,
                 dropout=0.05, lr=0.0005, batch_size=32, n_epochs=200,
                 weight_decay=0.01, device=None,
                 use_feature_modules=True,
                 model_type: str = 'transformer'):
        
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.weight_decay = weight_decay
        self.use_feature_modules = use_feature_modules
        
        # Model backend type: 'transformer' (default) or 'mamba'
        assert model_type in {'transformer', 'mamba'}, "model_type must be 'transformer' or 'mamba'"
        self.model_type = model_type
        
        # Device setup
        self.device = device if device is not None else DEVICE
        dprint(f"Using device: {self.device}")
        
        # Model will be initialized when data dimensions are known
        self.model = None
        
        # Mixed precision scaler for GPU
        self.scaler = GradScaler('cuda')
        
        # Use FeatureNormalizer instead of separate normalization parameters
        self.normalizer = FeatureNormalizer()
    
    def save_model(self, filepath, metadata=None):
        """
        Save the full model and configuration to file.
        
        Args:
            filepath: Path to save the model
            metadata: Optional dictionary with additional metadata to save
        """
        if metadata is None:
            metadata = {}
        # Record feature module configurations for ablation and reconstruction
        if self.use_feature_modules and hasattr(self, 'feature_modules'):
            fm_configs = []
            for mod in self.feature_modules:
                if isinstance(mod, CSPModule):
                    fm_configs.append({
                        'name': 'CSPModule',
                        'params': {
                            'n_components': mod.n_components,
                            'per_subject': mod.per_subject,
                            'filter_bank': mod.filter_bank,
                            'bands': mod.bands,
                            'sfreq': mod.sfreq
                        }
                    })
                elif isinstance(mod, SpectrogramModule):
                    fm_configs.append({
                        'name': 'SpectrogramModule',
                        'params': {
                            'n_channels': mod.n_channels,
                            'n_fft': mod.n_fft,
                            'hop_length': mod.hop_length,
                            'hidden': mod.hidden,
                            'out_dim': mod.out_dim,
                            'dropout': mod.dropout
                        }
                    })
                else:
                    raise ValueError(f'Unsupported feature module type: {type(mod)}')
            metadata['feature_modules'] = fm_configs
        
        # Save configuration and normalization to a separate file
        config_path = filepath.replace('.pt', '_config.pt')
        config_data = {
            'hyperparams': {
                'n_classes': self.n_classes,
                'embedding_dim': self.embedding_dim,
                'n_heads': self.n_heads,
                'n_layers': self.n_layers,
                'dropout': self.dropout,
                'use_feature_modules': self.use_feature_modules,
                'model_type': self.model_type,
            },
            'normalizer': self.normalizer,
            'metadata': metadata
        }
        
        # Save fitted feature modules state for online inference
        if self.use_feature_modules and hasattr(self, 'feature_modules'):
            config_data['saved_feature_modules'] = self.feature_modules
        
        # Save configuration to a separate file
        torch.save(config_data, config_path)
        dprint(f"Saved model configuration to {config_path}")
        
        # Save the full model (includes architecture and parameters)
        torch.save(self.model, filepath)
        dprint(f"Saved full model to {filepath}")
    
    @classmethod
    def load_model(cls, filepath, device=None):
        """
        Load a saved model and its configuration.
        
        Args:
            filepath: Path to the saved model
            device: Device to load the model to (None for automatic)
            
        Returns:
            classifier: Loaded classifier
            metadata: Additional metadata
        """
        # Set device
        if device is None:
            device = DEVICE

        # Load model and configuration from cache or disk
        if filepath in _load_model_cache:
            orig_model, config_data = _load_model_cache[filepath]
            dprint(f"Using cached model for {filepath}")
            # Deep copy to ensure independent model instance
            model = copy.deepcopy(orig_model).to(device)
        else:
            dprint(f"Loading model from {filepath}")
            model = torch.load(filepath, map_location=device, weights_only=False)
            # Load configuration file
            config_path = filepath.replace('.pt', '_config.pt')
            assert os.path.exists(config_path), f"Config file not found: {config_path}"
            dprint(f"Loading configuration from {config_path}")
            config_data = torch.load(config_path, map_location=device, weights_only=False)
            # Cache a pristine CPU copy of the model and its configuration.
            # CRITICAL: nn.Module.cpu()/.to() operate IN PLACE and return self,
            # so the cache must hold a deepcopy that is never handed to callers.
            # Caching `model.cpu()` directly aliases the caller's model: any
            # subsequent fine-tuning mutates the cached weights, and every later
            # load_model() of the same path returns already-trained weights
            # (cross-experiment contamination / test-set leakage).
            _load_model_cache[filepath] = (copy.deepcopy(model).cpu(), config_data)
            # Move model back to requested device
            model = model.to(device)

        # Extract configuration data
        hyperparams = config_data.get('hyperparams', {})
        normalizer = config_data.get('normalizer')
        metadata = config_data.get('metadata', {})
        dprint("Loaded configuration data")
        
        # Required parameters that must be present
        required_params = [
            'n_classes', 
            'embedding_dim',
            'n_heads',
            'n_layers',
            'dropout',
            'use_feature_modules'
        ]
        
        # If hyperparams is empty, try to extract from model
        if not hyperparams:
            dprint("Config file missing hyperparams section, extracting from model attributes")
            hyperparams = {}
            for param in required_params:
                try:
                    hyperparams[param] = getattr(model, param)
                    dprint(f"Extracted {param}={hyperparams[param]} from model attributes")
                except AttributeError:
                    raise ValueError(f"Critical parameter '{param}' not found in model attributes")
        
        # Backward-compatibility defaults for older checkpoints
        hyperparams.setdefault('use_feature_modules', True)
        hyperparams.setdefault('model_type', 'transformer')

        # Verify all required parameters are present
        missing_params = [param for param in required_params if param not in hyperparams]
        if missing_params:
            raise ValueError(f"Missing required parameters in config file: {missing_params}")
        
        # Create a new classifier with the loaded hyperparameters
        classifier = cls(
            n_classes=hyperparams['n_classes'],
            embedding_dim=hyperparams['embedding_dim'],
            n_heads=hyperparams['n_heads'],
            n_layers=hyperparams['n_layers'],
            dropout=hyperparams['dropout'],
            use_feature_modules=hyperparams['use_feature_modules'],
            model_type=hyperparams.get('model_type', 'transformer'),
            device=device
        )
        
        # Set the model
        classifier.model = model
        
        # Require normalizer to be present
        if normalizer is None:
            raise ValueError("Normalizer not found in config file. Cannot proceed with loading model.")
        
        classifier.normalizer = normalizer
        # Load saved feature modules for inference if available
        if 'saved_feature_modules' in config_data:
            classifier.feature_modules = config_data['saved_feature_modules']
            dprint("Loaded saved feature modules for inference")
        
        # Load global margin stats if available
        if 'margin_mean' in metadata and 'margin_std' in metadata:
            mmean = metadata['margin_mean']
            mstd  = metadata['margin_std']
            classifier.model.register_buffer('margin_mean', torch.tensor(mmean, device=device))
            classifier.model.register_buffer('margin_std',  torch.tensor(mstd, device=device))
        
        dprint("Loaded normalizer from config")
        dprint(f"Successfully loaded model from {filepath}")
        return classifier, metadata
    
    def evaluate(self, X_raw, y, subject_indices=None):
        """
        Evaluate the model on new data using the unified normalizer.
        """
        # Ensure model and normalizer are ready
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        assert self.normalizer is not None, "Normalizer must be initialized before evaluation"
        
        # Extract features via feature_modules if provided
        X_feat = None
        if hasattr(self, 'feature_modules') and self.feature_modules and self.use_feature_modules:
            feat_list = []
            for mod in self.feature_modules:
                feat = mod.transform(X_raw, subject_indices)
                # Ensure feat is tensor
                if not torch.is_tensor(feat):
                    feat = torch.tensor(feat, dtype=torch.float32)
                feat_list.append(feat.to(self.device))
            # Concatenate feature streams via torch
            X_feat = torch.cat(feat_list, dim=1)
            dprint(f"Extracted features: {X_feat.shape}")
        else:
            X_feat = None

        # Apply normalization to raw and feature data
        X_raw_norm, X_feat_norm = self.normalizer.transform(X_raw, X_feat)
        
        dataset = self._create_dataset(X_raw_norm, y, X_feat_norm)
        
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
        
        # Add progress bar for evaluation if DEBUG is on
        if DEBUG:
            eval_iter = tqdm(dataloader, desc="Evaluating", leave=False)
        else:
            eval_iter = dataloader
        
        with torch.no_grad():
            for batch in eval_iter:
                # Unpack raw and feature batch
                if len(batch) == 3:
                    X_raw_batch, X_feat_batch, y_batch = batch
                    feat_batch = X_feat_batch
                else:
                    # raw only
                    X_raw_batch, y_batch = batch
                    feat_batch = None
                # Forward pass with raw and feature streams
                outputs = self.model(X_raw_batch, feat_batch)
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
        cm = confusion_matrix(all_targets, all_preds)
        
        dprint(f"\nEvaluation results:")
        dprint(f"Accuracy: {accuracy:.4f}")
        dprint(f"Confusion matrix:\n{cm}")
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'targets': all_targets,
            'probabilities': all_probs
        }
    
    def ablation_report(self, X_raw, y):
        pass
    #     """
    #     Generate accuracy report when selectively disabling input modalities to assess their contribution.

    #     Args:
    #         X_raw: Raw EEG data array (n_samples, n_channels, n_times)
    #         y: True labels array (n_samples,)
    #         filter_bank: Boolean flag to indicate use of filter-bank CSP
    #     Returns:
    #         Dictionary mapping modalities to accuracy:
    #             'raw_only': accuracy using only raw time-domain data
    #             'csp_only': accuracy using only CSP features
    #             'full': accuracy using all inputs
    #     """
    #     if not hasattr(self, '_ablation_subject_indices'):
    #         raise ValueError("ablation_report requires setting self._ablation_subject_indices before calling")
    #     subject_indices = self._ablation_subject_indices
    #     # Build CSP transformer on this data
    #     if csp_transformer is not None:
    #         self.csp_transformer = csp_transformer
    #     else:
    #         assert self.use_csp is False

    #     # Save original flags
    #     orig_use_csp = self.use_csp
    #     report = {}
    #     # TODO: FIX
    #     # Raw only
    #     self.use_csp = False
    #     report['raw_only'] = self.evaluate(X_raw, y, subject_indices=subject_indices)['accuracy']
    #     # CSP only (drop raw input)
    #     X_zero = np.zeros_like(X_raw)
    #     self.use_csp = True
    #     report['csp_only'] = self.evaluate(X_zero, y, subject_indices=subject_indices)['accuracy']
    #     # Full inputs
    #     self.use_csp = orig_use_csp
    #     report['full'] = self.evaluate(X_raw, y, subject_indices=subject_indices)['accuracy']
    #     # Restore original flags
    #     self.use_csp = orig_use_csp
    #     return report
    
    def train_and_evaluate(self, X_train, y_train, X_val, y_val,
                           train_subject_indices=None, val_subject_indices=None,
                           early_stopping_patience=30,
                           fine_tuning=False, use_lr_scheduler=None, 
                           X_test=None, y_test=None, test_subject_indices=None):
        """
        Train and evaluate the hybrid model with early stopping
        
        Args:
            X_raw: Raw EEG data (pre-augmented if augmentation is desired)
            y: Labels corresponding to X_raw
            X_val: Optional pre-split validation data
            y_val: Optional pre-split validation labels
            val_split: Proportion of data for validation (used only if X_val/y_val not provided)
            early_stopping_patience: Number of epochs with no improvement before stopping
            fine_tuning: Whether this is a fine-tuning run
            use_lr_scheduler: Type of learning rate scheduler to use
            X_test: Optional test data to track test accuracy during training
            y_test: Optional test labels to track test accuracy during training
        """
        # Get data dimensions
        _, n_channels, n_times = X_train.shape
        
        dprint(f"Training class distribution: {torch.bincount(y_train)}")
        dprint(f"Validation class distribution: {torch.bincount(y_val)}")
        
        # 1. Feature extraction via feature_modules
        X_feat_train = None
        X_feat_val = None
        if hasattr(self, 'feature_modules') and self.feature_modules and self.use_feature_modules:
            feat_train_list = []
            feat_val_list = []
            for mod in self.feature_modules:
                feat_train = mod.transform(X_train, train_subject_indices)
                feat_val = mod.transform(X_val, val_subject_indices)
                feat_train_list.append(feat_train)
                feat_val_list.append(feat_val)
            # Concatenate feature streams as torch Tensors
            X_feat_train = torch.cat(feat_train_list, dim=1)
            X_feat_val = torch.cat(feat_val_list, dim=1)
            # Ensure on correct device
            X_feat_train = X_feat_train.to(self.device)
            X_feat_val = X_feat_val.to(self.device)
            dprint(f"Extracted features shape: training {X_feat_train.shape}, validation {X_feat_val.shape}")
        else:
            X_feat_train = None
            X_feat_val = None
        
        # 3. Initialize model if needed
        if not fine_tuning or self.model is None:
            dprint("Initializing new model...")
            # Select backbone implementation based on model_type
            if self.model_type == 'transformer':
                ModelClass = HybridCNNTransformer
            elif self.model_type == 'mamba':
                ModelClass = HybridCNNMamba
            else:
                raise ValueError(f"Unknown model_type '{self.model_type}'.")

            self.model = ModelClass(
                n_channels=n_channels,
                n_times=n_times,
                n_classes=self.n_classes,
                embedding_dim=self.embedding_dim,
                n_heads=self.n_heads,
                n_layers=self.n_layers,
                dropout=self.dropout,
                use_feature_modules=self.use_feature_modules,
                feature_dim=sum(mod.feature_dim for mod in self.feature_modules),
            ).to(self.device)
            
            # Print model summary
            dprint(f"Hybrid CNN/Transformer model architecture:\n{self.model}")
            total_params = sum(p.numel() for p in self.model.parameters())
            dprint(f"Total parameters: {total_params:,}")
        else:
            dprint(f"Using existing pre-trained model for fine-tuning")
            total_params = sum(p.numel() for p in self.model.parameters())
            dprint(f"Total parameters: {total_params:,}")
        
        # 4. Fit normalizer on original (non-augmented) raw data and extracted features
        if not fine_tuning or self.normalizer is None:
            dprint("Fitting new normalizer on original training data features...")
            self.normalizer = FeatureNormalizer()
            X_orig_feat = X_feat_train
            self.normalizer.fit(X_train, X_orig_feat)
        else:
            dprint("Using existing normalizer for fine-tuning")
        
        # 5. Apply normalization to raw and extracted features
        dprint("Applying normalization...")
        X_train_norm, X_feat_train_norm = self.normalizer.transform(X_train, X_feat_train)
        X_val_norm, X_feat_val_norm = self.normalizer.transform(X_val, X_feat_val)
        
        # Create fixed-length dataset (raw + features)
        train_dataset = self._create_dataset(X_train_norm, y_train, X_feat_train_norm)
        
        # Create standard data loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False
        )
            
        # Create validation dataset (raw + features)
        val_dataset = self._create_dataset(X_val_norm, y_val, X_feat_val_norm)
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Set up optimizer and loss function
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        # Class weights for imbalanced data - use training set distribution
        class_counts = torch.bincount(y_train).float()
        total = class_counts.sum()
        n_classes = class_counts.numel()
        class_weights = (total / (n_classes * class_counts)).sqrt().to(self.device)
        dprint(f"Class weights (from training set): {class_weights}")
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Learning rate scheduler
        if use_lr_scheduler == 'cosine':
            # Cosine annealing with warm restarts - modified for stability
            # Increase T_0 for less frequent restarts and higher min LR
            t_0 = max(25, min(75, self.n_epochs // 2))  # Longer first cycle
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=t_0, T_mult=1, eta_min=self.lr/10
            )
            dprint(f"Using cosine annealing scheduler with T_0={t_0}, min_lr={self.lr/10:.8f}")
        elif use_lr_scheduler == 'plateau':
            # Reduce on plateau (existing code)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10, verbose=True
            )
        elif use_lr_scheduler == 'onecycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.lr*5,
                steps_per_epoch=len(train_loader),
                epochs=self.n_epochs,
                pct_start=0.1
            )
        else:
            # No scheduler
            scheduler = None
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        # Early stopping variables
        best_val_loss = float('inf')
        best_acc = 0
        patience_counter = 0
        best_model_state = None
        
        dprint(f"Training for {self.n_epochs} epochs with early stopping (patience={early_stopping_patience})")
        
        try:
            # Create progress bar for epochs if DEBUG is on
            if not fine_tuning:
                epoch_iter = tqdm(range(self.n_epochs), desc="Training", position=0)
            else:
                epoch_iter = range(self.n_epochs)
            
            # Training loop
            for epoch in epoch_iter:
                # Training phase
                self.model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                # Create progress bar for batches if DEBUG is on
                if DEBUG and not fine_tuning:
                    batch_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.n_epochs} (Train)", position=1, leave=False)
                else:
                    batch_iter = train_loader
                    
                for batch in batch_iter:
                    # Unpack raw and feature batch
                    if len(batch) == 3:
                        X_raw_batch, X_feat_batch, y_batch = batch
                    else:
                        X_raw_batch, y_batch = batch
                        X_feat_batch = None
                    # Zero gradients
                    optimizer.zero_grad()
                    # Mixed precision forward and backward
                    with autocast('cuda'):
                        outputs = self.model(X_raw_batch, X_feat_batch)
                        loss = criterion(outputs, y_batch)
                    # Scale loss and backpropagate
                    self.scaler.scale(loss).backward()
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20.0)
                    # Step optimizer with scaled gradients
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    # Step scheduler if using onecycle
                    if use_lr_scheduler == 'onecycle':
                        scheduler.step()
                    
                    # Track statistics
                    train_loss += loss.item() * X_raw_batch.size(0)
                    _, predicted = torch.max(outputs, 1)
                    train_total += y_batch.size(0)
                    train_correct += (predicted == y_batch).sum().item()
                    
                    # Update progress bar if DEBUG and has set_postfix method
                    if DEBUG and hasattr(batch_iter, 'set_postfix'):
                        batch_iter.set_postfix(loss=f"{loss.item():.4f}", 
                                            acc=f"{(predicted == y_batch).sum().item() / y_batch.size(0):.4f}")
                
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
                
                # Create progress bar for validation if DEBUG is on
                if DEBUG and not fine_tuning:
                    val_iter = tqdm(val_loader, desc=f"Epoch {epoch+1}/{self.n_epochs} (Val)", position=1, leave=False)
                else:
                    val_iter = val_loader
                    
                with torch.no_grad():
                    for batch in val_iter:
                        # Unpack raw and feature batch
                        if len(batch) == 3:
                            X_raw_batch, X_feat_batch, y_batch = batch
                        else:
                            X_raw_batch, y_batch = batch
                        # Forward pass
                        outputs = self.model(X_raw_batch, X_feat_batch)
                        loss = criterion(outputs, y_batch)
                        
                        # Track statistics
                        val_loss += loss.item() * X_raw_batch.size(0)
                        _, predicted = torch.max(outputs, 1)
                        val_total += y_batch.size(0)
                        val_correct += (predicted == y_batch).sum().item()
                        
                        # Store predictions and targets for analysis
                        all_val_preds.extend(predicted.cpu().numpy())
                        all_val_targets.extend(y_batch.cpu().numpy())
                        
                        # Update progress bar if DEBUG and has set_postfix method
                        if DEBUG and hasattr(val_iter, 'set_postfix'):
                            val_iter.set_postfix(loss=f"{loss.item():.4f}", 
                                              acc=f"{(predicted == y_batch).sum().item() / y_batch.size(0):.4f}")
                
                # Calculate validation metrics
                val_loss = val_loss / val_total
                val_acc = val_correct / val_total
                
                # Test accuracy evaluation if test data is provided
                test_acc = None
                if X_test is not None and y_test is not None:
                    self.model.eval()
                    test_correct = 0
                    test_total = 0
                    with torch.no_grad():
                        # Test in smaller batches to avoid memory issues
                        test_batch_size = self.batch_size
                        num_test_batches = int(np.ceil(len(X_test) / test_batch_size))
                        
                        for i in range(num_test_batches):
                            start_idx = i * test_batch_size
                            end_idx = min((i + 1) * test_batch_size, len(X_test))
                            
                            # Extract batch
                            X_test_batch = X_test[start_idx:end_idx]
                            y_test_batch = y_test[start_idx:end_idx]
                            
                            # Get feature modules output if needed
                            # (subject indices must be sliced to match the batch)
                            test_subj_batch = test_subject_indices[start_idx:end_idx] if test_subject_indices is not None else None
                            X_test_feat_batch = None
                            if hasattr(self, 'feature_modules') and self.feature_modules and self.use_feature_modules:
                                feat_test_list = []
                                for mod in self.feature_modules:
                                    feat_test = mod.transform(X_test_batch, test_subj_batch)
                                    if not torch.is_tensor(feat_test):
                                        feat_test = torch.tensor(feat_test, dtype=torch.float32)
                                    feat_test_list.append(feat_test)
                                X_test_feat_batch = torch.cat(feat_test_list, dim=1)
                            
                            # Apply normalization
                            X_test_norm, X_test_feat_norm = self.normalizer.transform(X_test_batch, X_test_feat_batch)
                            
                            # Convert normalized data to torch tensors on the correct device
                            X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32, device=self.device)
                            # labels are already tensors on device
                            y_test_tensor = y_test_batch
                            if self.use_feature_modules:
                                X_test_feat_tensor = torch.tensor(X_test_feat_norm, dtype=torch.float32, device=self.device)
                            else:
                                X_test_feat_tensor = None
                            # Forward pass
                            outputs = self.model(X_test_tensor, X_test_feat_tensor)
                            _, predicted = torch.max(outputs, 1)
                            
                            # Update stats
                            test_total += y_test_batch.size(0)
                            test_correct += (predicted == y_test_batch).sum().item()
                    
                    # Calculate test accuracy
                    test_acc = test_correct / test_total
                    # dprint(f"Test accuracy at epoch {epoch+1}: {test_acc:.4f}")
                
                # Update learning rate scheduler
                if use_lr_scheduler == 'cosine':
                    scheduler.step()  # Cosine scheduler steps every epoch
                elif use_lr_scheduler == 'plateau':
                    scheduler.step(val_loss)  # Plateau scheduler uses validation loss
                
                # Update history
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['train_acc'].append(train_acc)
                history['val_acc'].append(val_acc)
                if test_acc is not None:
                    if 'test_acc' not in history:
                        history['test_acc'] = []
                    history['test_acc'].append(test_acc)
                
                if DEBUG and hasattr(epoch_iter, 'set_postfix'):
                    # Update progress bar
                    epoch_iter.set_postfix(
                        train_loss=f"{train_loss:.4f}", 
                        train_acc=f"{train_acc:.4f}",
                        val_loss=f"{val_loss:.4f}", 
                        val_acc=f"{val_acc:.4f}"
                    )
                
                # Check for improvement
                improvement = ''
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Use deepcopy to ensure we get a completely independent copy of the state dict
                    best_model_state = deepcopy(self.model.state_dict())
                    # Store best validation predictions for consistency check
                    self._best_val_preds = all_val_preds.copy()
                    self._best_val_targets = all_val_targets.copy()
                    improvement = '✓'
                else:
                    patience_counter += 1
                    improvement = f"× ({patience_counter}/{early_stopping_patience})"
                
                if epoch % 10 == 0:
                    dprint(f"Epoch {epoch+1}/{self.n_epochs} - "
                        f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
                        f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f} {improvement}")
                    
                # Early stopping
                if patience_counter >= early_stopping_patience:
                    if epoch % 10 != 0:
                        dprint(f"Epoch {epoch+1}/{self.n_epochs} - "
                            f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
                            f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f} {improvement}")
                    dprint(f"Early stopping at epoch {epoch+1}")
                    if DEBUG:
                        try:
                            epoch_iter.close()
                        except:
                            pass
                    break
        
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user at epoch {}/{}".format(epoch+1, self.n_epochs))
            print("Evaluating best model found so far...")
            if DEBUG:
                try:
                    epoch_iter.close()
                except:
                    pass
        
        # Load best model
        if best_model_state is not None:
            dprint(f"Loading best model state with accuracy: {best_acc:.4f}")
            # Check the state_dict size before loading
            params_before = {}
            for name, param in self.model.named_parameters():
                params_before[name] = param.data.clone()
            
            # Force one small change to verify we can detect parameter changes
            if next(self.model.parameters()).requires_grad:
                first_param = next(iter(self.model.parameters()))
                first_param.data[0, 0] += 0.01  # Small modification to verify change detection
            
            # Load the state dict
            self.model.load_state_dict(best_model_state)
            
            # Verify the model was loaded correctly by checking if parameters changed
            params_unchanged = 0
            params_changed = 0
            for name, param in self.model.named_parameters():
                if torch.all(torch.eq(param.data, params_before[name])):
                    params_unchanged += 1
                else:
                    params_changed += 1
            
            dprint(f"Model state loaded: {params_changed} parameters changed, {params_unchanged} unchanged")
            
            # If no parameters changed, this is a critical error
            assert params_changed != 0, "No parameters changed when loading the best model state!"
        
        # Evaluate best model on validation set
        dprint("Evaluating best model on validation set...")
        eval_results = self.evaluate(X_val, y_val, val_subject_indices)
        
        # Add history to results
        eval_results['history'] = history
        
        return eval_results
    
    def _create_dataset(self, X_raw, y, X_feat=None):
        """Helper method to create a PyTorch dataset (raw + feature vectors), staying in torch."""
        # Raw input
        if torch.is_tensor(X_raw):
            X_raw_tensor = X_raw.float().to(self.device)
        else:
            X_raw_tensor = torch.tensor(X_raw, dtype=torch.float32, device=self.device)
        # Labels
        if torch.is_tensor(y):
            y_tensor = y.long().to(self.device)
        else:
            y_tensor = torch.tensor(y, dtype=torch.int64, device=self.device)
        if X_feat is not None:
            if torch.is_tensor(X_feat):
                X_feat_tensor = X_feat.float().to(self.device)
            else:
                X_feat_tensor = torch.tensor(X_feat, dtype=torch.float32, device=self.device)
            dataset = TensorDataset(X_raw_tensor, X_feat_tensor, y_tensor)
        else:
            dataset = TensorDataset(X_raw_tensor, y_tensor)
        return dataset
    
    def freeze_layers(self, unfrozen_layers=['classifier']):
        """
        Freezes model layers for fine-tuning
        Args:
            unfrozen_layers (list): List of layer names to keep unfrozen (e.g. ['classifier, transformer_layers, cnn_encoder'])
        """
        if self.model is None:
            raise ValueError("Model must be initialized before freezing layers")
            
        # # Print model structure to debug attribute names
        # print("Model structure:")
        # for name, _ in self.model.named_children():
        #     print(f"  - {name}")
        
         # First freeze ALL parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Assert that all layers to be unfrozen exist
        for layer in unfrozen_layers:
            assert hasattr(self.model, layer), f"Model does not have a '{layer}' attribute!"
            for param in getattr(self.model, layer).parameters():
                param.requires_grad = True

        if DEBUG:
            # Count frozen vs trainable parameters
            total_params = 0
            trainable_params = 0
            
            for name, param in self.model.named_parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
                    # dprint(f"Trainable: {name} ({param.numel()} parameters)")
                    
            dprint(f"Froze all layers except classifier.")
            dprint(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")