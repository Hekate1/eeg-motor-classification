"""
Model classifier module for EEG classification.

This module provides the classifier class for training and evaluating models.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix
import time
import math
import matplotlib.pyplot as plt
import multiprocessing

from models.data_augmenter import OptimizedAugmenter
from models.hybrid_cnn_transformer import HybridCNNTransformer


def subject_specific_normalize(X):
    """
    Normalize EEG data separately for each subject trial
    
    Args:
        X: EEG data with shape (n_trials, n_channels, n_times)
        
    Returns:
        X_norm: Normalized EEG data
    """
    X_norm = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        # For each trial, normalize each channel to zero mean and unit variance
        for c in range(X.shape[1]):
            channel_data = X[i, c, :]
            channel_mean = np.mean(channel_data)
            channel_std = np.std(channel_data)
            if channel_std > 0:  # Avoid division by zero
                X_norm[i, c, :] = (channel_data - channel_mean) / channel_std
            else:
                X_norm[i, c, :] = channel_data - channel_mean
                
    return X_norm


class HybridModelClassifier:
    """
    Wrapper class for training and evaluating the Hybrid CNN/Transformer model
    """
    def __init__(self, n_classes, embedding_dim=128, n_heads=4, n_layers=2,
                dropout=0.3, lr=0.0005, batch_size=32, n_epochs=200,
                weight_decay=0.01, device=None, augmentation_factor=2,
                use_csp=True, use_freq=True, n_csp_components=4,
                force_balanced_val=False, random_state=42, num_workers=None):
        
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
        self.force_balanced_val = force_balanced_val
        self.random_state = random_state
        
        # Set number of workers for data loading (default: CPU count or 4, whichever is smaller)
        if num_workers is None:
            # Use more workers to better utilize CPU cores
            self.num_workers = min(multiprocessing.cpu_count(), 8)  # Increased from 4 to 8
        else:
            self.num_workers = num_workers
        print(f"Using {self.num_workers} workers for data loading")
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Initialize augmenter
        self.augmenter = OptimizedAugmenter()
        
        # Initialize model and other components to None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.history = None
        
    def _create_model(self, n_channels, n_times, n_freq_features=None):
        """
        Create and initialize the model
        """
        model = HybridCNNTransformer(
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
        
        return model
    
    def _create_dataloaders(self, X_raw, y, X_csp=None, X_freq=None, val_split=0.2, 
                          train_indices=None, val_indices=None, apply_augmentation=False):
        """
        Create PyTorch dataloaders for training and validation.
        
        Args:
            X_raw: Raw EEG data [n_trials, n_channels, n_times]
            y: Labels [n_trials]
            X_csp: CSP features [n_trials, n_components]
            X_freq: Frequency features [n_trials, n_features]
            val_split: Validation split ratio
            train_indices: Indices for training set (if None, will be created based on val_split)
            val_indices: Indices for validation set (if None, will be created based on val_split)
            apply_augmentation: Whether to apply data augmentation
            
        Returns:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            train_indices: Indices used for training
            val_indices: Indices used for validation
        """
        # Create train/val split if not provided
        if train_indices is None or val_indices is None:
            # Stratified split
            from sklearn.model_selection import train_test_split
            train_indices, val_indices = train_test_split(
                np.arange(len(y)), 
                test_size=val_split,
                stratify=y,
                random_state=self.random_state
            )
        
        # Get training and validation data
        X_raw_train = X_raw[train_indices]
        y_train = y[train_indices]
        
        X_raw_val = X_raw[val_indices]
        y_val = y[val_indices]
        
        # Handle CSP features
        if X_csp is not None:
            X_csp_train = X_csp[train_indices]
            X_csp_val = X_csp[val_indices]
        else:
            X_csp_train = None
            X_csp_val = None
            
        # Handle frequency features
        if X_freq is not None:
            X_freq_train = X_freq[train_indices]
            X_freq_val = X_freq[val_indices]
        else:
            X_freq_train = None
            X_freq_val = None
        
        # Apply data augmentation for training set if requested
        if apply_augmentation and self.augmentation_factor > 0:
            from models.evaluation import extract_robust_csp_features, extract_frequency_features, BANDS
            
            # Augment training data
            X_raw_train_aug, y_train_aug = self._augment_data(X_raw_train, y_train)
            print(f"Generated {len(X_raw_train_aug)} samples from {len(X_raw_train)} original samples")
            
            # If we augmented the data, we need to recalculate the CSP and frequency features
            # instead of duplicating them, to ensure proper alignment with augmented data
            if X_csp is not None:
                print("Recalculating CSP features for augmented data...")
                try:
                    X_csp_train_aug, _ = extract_robust_csp_features(
                        X_raw_train_aug, y_train_aug, 
                        n_components=X_csp_train.shape[1],
                        reg=0.1
                    )
                    X_csp_train = X_csp_train_aug
                except Exception as e:
                    print(f"Error recalculating CSP features: {e}")
                    print("Duplicating CSP features to match augmented data")
                    # Duplicate features as fallback (old behavior)
                    X_csp_train = self._duplicate_features(X_csp_train, len(X_raw_train_aug))
            
            if X_freq is not None:
                print("Recalculating frequency features for augmented data...")
                try:
                    X_freq_train_aug = extract_frequency_features(X_raw_train_aug, BANDS)
                    X_freq_train = X_freq_train_aug
                except Exception as e:
                    print(f"Error recalculating frequency features: {e}")
                    print("Duplicating frequency features to match augmented data")
                    # Duplicate features as fallback (old behavior)
                    X_freq_train = self._duplicate_features(X_freq_train, len(X_raw_train_aug))
            
            # Replace training data with augmented data
            X_raw_train = X_raw_train_aug
            y_train = y_train_aug
        
        # Convert data to PyTorch tensors
        X_raw_train_tensor = torch.FloatTensor(X_raw_train)
        y_train_tensor = torch.LongTensor(y_train)
        
        X_raw_val_tensor = torch.FloatTensor(X_raw_val)
        y_val_tensor = torch.LongTensor(y_val)
        
        # Create datasets
        train_dataset = self._create_dataset(X_raw_train_tensor, y_train_tensor, X_csp_train, X_freq_train)
        val_dataset = self._create_dataset(X_raw_val_tensor, y_val_tensor, X_csp_val, X_freq_val)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers
        )
        
        return train_loader, val_loader, train_indices, val_indices
        
    def _duplicate_features(self, features, target_size):
        """
        Duplicate features to match target size (used as fallback method)
        """
        if features is None:
            return None
            
        # Check if we need to duplicate
        if len(features) >= target_size:
            return features[:target_size]
            
        # Calculate repetition factor
        repeat_factor = int(np.ceil(target_size / len(features)))
        
        # Duplicate features
        duplicated = np.tile(features, (repeat_factor, 1))
        
        # Trim to exact size
        return duplicated[:target_size]
    
    def _forward_pass(self, batch, force_balanced=False):
        """
        Forward pass through the model based on available features
        """
        if len(batch) == 4:  # Raw + CSP + Freq + labels
            x_raw, x_csp, x_freq, y = batch
            x_raw = x_raw.to(self.device)
            x_csp = x_csp.to(self.device)
            x_freq = x_freq.to(self.device)
            y = y.to(self.device)
            outputs = self.model(x_raw, x_csp, x_freq, force_balanced=force_balanced)
        elif len(batch) == 3:  # Raw + CSP/Freq + labels
            if self.use_csp and not self.use_freq:
                x_raw, x_csp, y = batch
                x_raw = x_raw.to(self.device)
                x_csp = x_csp.to(self.device)
                y = y.to(self.device)
                outputs = self.model(x_raw, x_csp, force_balanced=force_balanced)
            elif self.use_freq and not self.use_csp:
                x_raw, x_freq, y = batch
                x_raw = x_raw.to(self.device)
                x_freq = x_freq.to(self.device)
                y = y.to(self.device)
                outputs = self.model(x_raw, None, x_freq, force_balanced=force_balanced)
            else:
                raise ValueError("Invalid batch structure for model configuration")
        elif len(batch) == 2:  # Raw + labels
            x_raw, y = batch
            x_raw = x_raw.to(self.device)
            y = y.to(self.device)
            outputs = self.model(x_raw, force_balanced=force_balanced)
        else:
            raise ValueError(f"Invalid batch size: {len(batch)}")
            
        return outputs, y
        
    def _initialize_training(self, class_weights=None):
        """
        Initialize training components: loss function, optimizer, and scheduler
        """
        # Use weighted cross-entropy if class weights are provided
        if class_weights is not None:
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
            print(f"Using weighted cross-entropy loss with weights: {class_weights}")
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        # Initialize optimizer with weight decay for regularization
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        # Initialize learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=15,
            min_lr=1e-6, verbose=True
        )
    
    def train_and_evaluate(self, X_raw, y, X_csp=None, X_freq=None, val_split=0.2,
                            train_indices=None, val_indices=None, 
                            apply_augmentation=True, custom_train_fn=None):
        """
        Train the model and evaluate on validation data.
        
        Args:
            X_raw: Raw EEG data [n_trials, n_channels, n_times]
            y: Labels [n_trials]
            X_csp: CSP features [n_trials, n_components]
            X_freq: Frequency features [n_trials, n_features]
            val_split: Validation split ratio
            train_indices: Indices for training set
            val_indices: Indices for validation set
            apply_augmentation: Whether to apply data augmentation
            custom_train_fn: Custom training function to use instead of default
            
        Returns:
            Dictionary with training results
        """
        if custom_train_fn is not None:
            # Use custom training function (useful for fine-tuning)
            return custom_train_fn(X_raw, y, X_csp, X_freq, val_split)
        
        # Create dataloaders
        train_loader, val_loader, train_idx, val_idx = self._create_dataloaders(
            X_raw, y, X_csp, X_freq, val_split, 
            train_indices, val_indices, 
            apply_augmentation=apply_augmentation
        )
        
        # Calculate class weights for handling class imbalance
        class_counts = np.bincount(y[train_idx])
        n_samples = len(train_idx)
        class_weights = n_samples / (len(class_counts) * class_counts)
        class_weights = class_weights / np.sum(class_weights) * len(class_counts)
        
        # Initialize training components
        self._initialize_training(class_weights)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_balanced_acc': []
        }
        
        # Best model tracking
        best_val_metric = 0.0
        best_epoch = 0
        patience = max(30, self.n_epochs // 10)  # 10% of total epochs or at least 30
        patience_counter = 0
        
        # Start training
        start_time = time.time()
        for epoch in range(self.n_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # Iterate over training batches
            for batch in train_loader:
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs, y_batch = self._forward_pass(batch)
                
                # Calculate loss
                loss = self.criterion(outputs, y_batch)
                
                # Backward pass and optimization
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # Update statistics
                train_loss += loss.item() * y_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == y_batch).sum().item()
                train_total += y_batch.size(0)
            
            # Calculate training metrics
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for batch in val_loader:
                    # Forward pass with and without balanced forcing
                    outputs, y_batch = self._forward_pass(batch, force_balanced=self.force_balanced_val)
                    
                    # Calculate loss
                    loss = self.criterion(outputs, y_batch)
                    
                    # Update statistics
                    val_loss += loss.item() * y_batch.size(0)
                    
                    # Store predictions and targets
                    _, predicted = torch.max(outputs, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(y_batch.cpu().numpy())
            
            # Calculate validation metrics
            val_loss = val_loss / len(val_loader.dataset)
            val_acc = accuracy_score(all_targets, all_preds)
            val_balanced_acc = balanced_accuracy_score(all_targets, all_preds)
            
            # Update learning rate scheduler
            self.scheduler.step(val_balanced_acc)
            
            # Save history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['val_balanced_acc'].append(val_balanced_acc)
            
            # Print progress
            if (epoch + 1) % max(1, self.n_epochs // 20) == 0:
                print(f"Epoch {epoch+1}/{self.n_epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Train Acc: {train_acc:.4f} | "
                      f"Val Acc: {val_acc:.4f} | "
                      f"Val Balanced Acc: {val_balanced_acc:.4f}")
            
            # Check for improvement
            current_metric = val_balanced_acc
            if current_metric > best_val_metric:
                best_val_metric = current_metric
                best_epoch = epoch
                patience_counter = 0
                
                # Save best model
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}. Best epoch: {best_epoch+1} with validation balanced accuracy: {best_val_metric:.4f}")
                break
        
        # Load best model
        self.model.load_state_dict(best_model_state)
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Final evaluation
        all_preds = []
        all_targets = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in val_loader:
                # Forward pass
                outputs, y_batch = self._forward_pass(batch)
                
                # Get predictions
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
        
        # Calculate final metrics
        final_val_acc = accuracy_score(all_targets, all_preds)
        final_val_balanced_acc = balanced_accuracy_score(all_targets, all_preds)
        conf_matrix = confusion_matrix(all_targets, all_preds)
        
        # Create output dictionary
        results = {
            'model': self.model,
            'history': history,
            'best_epoch': best_epoch,
            'val_acc': final_val_acc,
            'val_balanced_acc': final_val_balanced_acc,
            'conf_matrix': conf_matrix,
            'training_time': training_time,
            'train_indices': train_idx,
            'val_indices': val_idx
        }
        
        print(f"Training completed. Final validation accuracy: {final_val_acc:.4f}, balanced accuracy: {final_val_balanced_acc:.4f}")
        print(f"Training time: {training_time:.2f} seconds")
        print("Confusion matrix:")
        print(conf_matrix)
        
        # Save history for later use
        self.history = history
        
        return results
    
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
        if subset == 'validation' and val_indices is not None:
            X_raw = X_raw[val_indices]
            y = y[val_indices]
            if X_csp is not None:
                X_csp = X_csp[val_indices]
            if X_freq is not None:
                X_freq = X_freq[val_indices]
        
        # Convert to PyTorch tensors
        X_raw_tensor = torch.tensor(X_raw, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        # Create dataset and loader
        if X_csp is not None and X_freq is not None:
            # Use both CSP and frequency features
            X_csp_tensor = torch.tensor(X_csp, dtype=torch.float32)
            X_freq_tensor = torch.tensor(X_freq, dtype=torch.float32)
            dataset = TensorDataset(X_raw_tensor, X_csp_tensor, X_freq_tensor, y_tensor)
        elif X_csp is not None:
            # Use only CSP features
            X_csp_tensor = torch.tensor(X_csp, dtype=torch.float32)
            dataset = TensorDataset(X_raw_tensor, X_csp_tensor, y_tensor)
        elif X_freq is not None:
            # Use only frequency features
            X_freq_tensor = torch.tensor(X_freq, dtype=torch.float32)
            dataset = TensorDataset(X_raw_tensor, X_freq_tensor, y_tensor)
        else:
            # Use only raw EEG
            dataset = TensorDataset(X_raw_tensor, y_tensor)
        
        # Create data loader
        batch_size = min(self.batch_size, len(dataset))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers)
        
        # Evaluate with and without balanced forcing
        all_results = {}
        
        for use_balanced in [False, True]:
            name_suffix = "_balanced" if use_balanced else ""
            
            # Evaluation phase
            self.model.eval()
            eval_loss = 0.0
            all_preds = []
            all_targets = []
            all_probs = []
            
            with torch.no_grad():
                for batch in loader:
                    # Forward pass
                    outputs, y_batch = self._forward_pass(batch, force_balanced=use_balanced)
                    
                    # Calculate loss
                    loss = nn.CrossEntropyLoss()(outputs, y_batch)
                    
                    # Update statistics
                    eval_loss += loss.item() * y_batch.size(0)
                    
                    # Store predictions and targets
                    probs = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(y_batch.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
            
            # Calculate evaluation metrics
            eval_loss = eval_loss / len(loader.dataset)
            eval_acc = accuracy_score(all_targets, all_preds)
            eval_balanced_acc = balanced_accuracy_score(all_targets, all_preds)
            conf_matrix = confusion_matrix(all_targets, all_preds)
            
            # Create output dictionary
            results = {
                f'loss{name_suffix}': eval_loss,
                f'accuracy{name_suffix}': eval_acc,
                f'balanced_accuracy{name_suffix}': eval_balanced_acc,
                f'confusion_matrix{name_suffix}': conf_matrix,
                f'predictions{name_suffix}': np.array(all_preds),
                f'probabilities{name_suffix}': np.array(all_probs),
                f'targets': np.array(all_targets)
            }
            
            all_results.update(results)
            
            print(f"Evaluation{' with balanced prediction' if use_balanced else ''}:")
            print(f"  Loss: {eval_loss:.4f}")
            print(f"  Accuracy: {eval_acc:.4f}")
            print(f"  Balanced Accuracy: {eval_balanced_acc:.4f}")
            print(f"  Confusion Matrix:\n{conf_matrix}")
        
        return all_results
    
    def save_model(self, save_path):
        """
        Save the trained model
        """
        if self.model is None:
            raise ValueError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Create save dictionary
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
            'history': self.history
        }
        
        # Save model
        torch.save(save_dict, save_path)
        print(f"Model saved to {save_path}")
        
    def load_model(self, load_path, n_channels, n_times, n_freq_features=None):
        """
        Load a trained model
        """
        # Load saved dictionary
        if torch.cuda.is_available():
            checkpoint = torch.load(load_path)
        else:
            checkpoint = torch.load(load_path, map_location=torch.device('cpu'))
        
        # Get model configuration
        model_config = checkpoint['model_config']
        self.n_classes = model_config['n_classes']
        self.embedding_dim = model_config.get('embedding_dim', self.embedding_dim)
        self.n_heads = model_config.get('n_heads', self.n_heads)
        self.n_layers = model_config.get('n_layers', self.n_layers)
        self.dropout = model_config.get('dropout', self.dropout)
        self.use_csp = model_config.get('use_csp', self.use_csp)
        self.use_freq = model_config.get('use_freq', self.use_freq)
        self.n_csp_components = model_config.get('n_csp_components', self.n_csp_components)
        
        # Create model
        self.model = self._create_model(n_channels, n_times, n_freq_features)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', None)
        
        print(f"Model loaded from {load_path}")
        
        return self.model
        
    def plot_training_curves(self, figsize=(12, 8), save_path=None):
        """
        Plot training and validation curves
        """
        if self.history is None:
            raise ValueError("No training history available")
        
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        
        # Plot loss
        axs[0, 0].plot(self.history['train_loss'], label='Training Loss')
        axs[0, 0].plot(self.history['val_loss'], label='Validation Loss')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].set_title('Training and Validation Loss')
        axs[0, 0].legend()
        
        # Plot accuracy
        axs[0, 1].plot(self.history['train_acc'], label='Training Accuracy')
        axs[0, 1].plot(self.history['val_acc'], label='Validation Accuracy')
        axs[0, 1].set_xlabel('Epoch')
        axs[0, 1].set_ylabel('Accuracy')
        axs[0, 1].set_title('Training and Validation Accuracy')
        axs[0, 1].legend()
        
        # Plot balanced accuracy
        axs[1, 0].plot(self.history['val_balanced_acc'], label='Validation Balanced Accuracy')
        axs[1, 0].set_xlabel('Epoch')
        axs[1, 0].set_ylabel('Balanced Accuracy')
        axs[1, 0].set_title('Validation Balanced Accuracy')
        axs[1, 0].legend()
        
        # Make the last subplot blank
        axs[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path is not None:
            plt.savefig(save_path)
            print(f"Training curves saved to {save_path}")
            
        return fig, axs 

    def _augment_data(self, X_raw, y):
        """
        Apply data augmentation to raw EEG data.
        
        Args:
            X_raw: Raw EEG data [n_trials, n_channels, n_times]
            y: Labels [n_trials]
            
        Returns:
            X_raw_aug: Augmented EEG data
            y_aug: Corresponding labels
        """
        # Set random seed for reproducibility
        np.random.seed(self.random_state)
        
        # Start with original data
        X_aug = X_raw.copy()
        y_aug = y.copy()
        
        # Number of original samples
        n_original = len(X_raw)
        
        # Number of samples to generate for each augmentation technique
        n_per_aug = int(n_original * self.augmentation_factor / 3)
        
        # 1. Add noise
        for _ in range(n_per_aug):
            # Randomly select a sample to augment
            idx = np.random.randint(0, n_original)
            
            # Apply noise augmentation
            noise_level = np.random.uniform(0.05, 0.2)
            noise = np.random.normal(0, noise_level, X_raw[idx].shape)
            augmented = X_raw[idx] + noise
            
            # Add to augmented dataset
            X_aug = np.vstack([X_aug, augmented[np.newaxis, :]])
            y_aug = np.append(y_aug, y[idx])
        
        # 2. Time warping
        for _ in range(n_per_aug):
            # Randomly select a sample to augment
            idx = np.random.randint(0, n_original)
            
            # Apply time warping (stretch/compress signal)
            stretch_factor = np.random.uniform(0.8, 1.2)
            n_channels, n_times = X_raw[idx].shape
            warped = np.zeros_like(X_raw[idx])
            
            for c in range(n_channels):
                # Interpolate the signal
                x = np.linspace(0, 1, n_times)
                x_warped = np.linspace(0, 1, int(n_times * stretch_factor))
                
                # Ensure x_warped doesn't exceed original range
                x_warped = np.clip(x_warped, 0, 1)
                
                # Interpolate signal
                from scipy.interpolate import interp1d
                f = interp1d(x, X_raw[idx, c, :], kind='linear', bounds_error=False, fill_value='extrapolate')
                warped_signal = f(x_warped)
                
                # Resize back to original size
                if len(warped_signal) != n_times:
                    warped_signal = np.interp(x, x_warped, warped_signal)
                
                warped[c, :] = warped_signal
            
            # Add to augmented dataset
            X_aug = np.vstack([X_aug, warped[np.newaxis, :]])
            y_aug = np.append(y_aug, y[idx])
        
        # 3. Amplitude scaling
        for _ in range(n_per_aug):
            # Randomly select a sample to augment
            idx = np.random.randint(0, n_original)
            
            # Apply amplitude scaling
            scale_factor = np.random.uniform(0.7, 1.3)
            scaled = X_raw[idx] * scale_factor
            
            # Add to augmented dataset
            X_aug = np.vstack([X_aug, scaled[np.newaxis, :]])
            y_aug = np.append(y_aug, y[idx])
        
        return X_aug, y_aug
    
    def _create_dataset(self, X_raw, y, X_csp=None, X_freq=None):
        """
        Create a PyTorch dataset with the given data.
        
        Args:
            X_raw: Raw EEG data tensor
            y: Labels tensor
            X_csp: CSP features (numpy array or None)
            X_freq: Frequency features (numpy array or None)
            
        Returns:
            PyTorch dataset
        """
        # Convert features to tensors if provided
        if X_csp is not None:
            X_csp = torch.FloatTensor(X_csp)
        
        if X_freq is not None:
            X_freq = torch.FloatTensor(X_freq)
        
        # Create appropriate dataset based on available features
        if X_csp is not None and X_freq is not None:
            # Use both CSP and frequency features
            return TensorDataset(X_raw, X_csp, X_freq, y)
        elif X_csp is not None:
            # Use only CSP features
            return TensorDataset(X_raw, X_csp, y)
        elif X_freq is not None:
            # Use only frequency features
            return TensorDataset(X_raw, X_freq, y)
        else:
            # Use only raw EEG
            return TensorDataset(X_raw, y) 