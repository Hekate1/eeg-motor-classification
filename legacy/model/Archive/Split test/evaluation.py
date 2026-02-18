"""
Evaluation module for EEG classification.

This module provides functions for model evaluation, cross-validation, 
and subject-specific fine-tuning.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score
from sklearn.model_selection import KFold
import time
from scipy import signal
import json
import glob
import re
import mne
from mne.decoding import CSP

from models.model_classifier import HybridModelClassifier
from models.hybrid_cnn_transformer import extract_csp_features, extract_frequency_features

# Constants
BANDS = [(8, 12), (13, 30)]  # Mu and Beta bands
RUN_IDS = ['3', '7', '11']  # Standard run IDs for CV


def extract_robust_csp_features(X, y, n_components=4, reg=0.1):
    """
    Extract CSP features with regularization to improve numerical stability.
    
    Args:
        X: EEG data [n_trials, n_channels, n_times]
        y: Labels [n_trials]
        n_components: Number of CSP components
        reg: Regularization parameter (higher = more stable but less accurate)
        
    Returns:
        X_csp: CSP features
        csp: Fitted CSP transformer
    """
    # Apply regularization to avoid numerical instability
    csp = CSP(n_components=n_components, reg=reg, log=True, norm_trace=False)
    
    # Clean data to remove any potential NaN/Inf values
    X_clean = np.copy(X)
    X_clean = np.nan_to_num(X_clean, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Fit and transform
    X_csp = csp.fit_transform(X_clean, y)
    
    return X_csp, csp


def load_and_prepare_eeg_data(subject_ids, n_csp_components=4):
    """
    Load preprocessed EEG data for multiple subjects from the processed_data directory.
    
    Args:
        subject_ids: List of subject IDs to load
        n_csp_components: Number of CSP components to use
        
    Returns:
        Dictionary with subject IDs as keys and their data as values
    """
    data = {}
    run_ids = [3, 7, 11]  # Motor imagery runs
    
    for subject_id in subject_ids:
        X_raw_all = []
        y_all = []
        
        for run_id in run_ids:
            try:
                # Construct path to the processed data file
                processed_file = f"processed_data/sub-{subject_id}_run-{run_id}_processed-epo.fif"
                
                if not os.path.exists(processed_file):
                    print(f"Processed file not found: {processed_file}")
                    continue
                
                print(f"Loading processed data from {processed_file}")
                
                # Load the processed epochs
                try:
                    epochs = mne.read_epochs(processed_file, preload=True)
                    print(f"Successfully loaded {processed_file}")
                    
                    # Get data and labels
                    X = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
                    y = epochs.events[:, 2] - 1  # Convert to 0-based indexing
                    
                    # Check for NaN or infinite values
                    if np.isnan(X).any() or np.isinf(X).any():
                        print(f"Warning: {processed_file} contains NaN or inf values. Attempting to clean...")
                        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                        # Check again after cleaning
                        if np.isnan(X).any() or np.isinf(X).any():
                            print(f"Error: Could not clean NaN/inf values in {processed_file}. Skipping.")
                            continue
                    
                    X_raw_all.append(X)
                    y_all.append(y)
                    
                    print(f"Added {len(y)} trials from {processed_file}")
                    
                except Exception as e:
                    print(f"Error loading {processed_file}: {str(e)}")
                    continue
                
            except Exception as e:
                print(f"Error processing subject {subject_id}, run {run_id}: {str(e)}")
                continue
        
        if len(X_raw_all) == 0:
            print(f"No data loaded for subject {subject_id}")
            continue
        
        # Combine data from all runs
        X_raw = np.concatenate(X_raw_all, axis=0)
        y = np.concatenate(y_all, axis=0)
        
        print(f"Total trials for subject {subject_id}: {len(y)}")
        
        # Check for class imbalance that would break CSP
        unique_classes, counts = np.unique(y, return_counts=True)
        if len(unique_classes) < 2:
            print(f"Warning: Subject {subject_id} has only {len(unique_classes)} class. CSP requires at least 2 classes.")
            X_csp, csp_transformer = None, None
        elif np.min(counts) < 5:
            print(f"Warning: Subject {subject_id} has class with < 5 trials ({counts}). CSP might be unstable.")
            # Try with increased regularization
            try:
                X_csp, csp_transformer = extract_robust_csp_features(X_raw, y, n_components=n_csp_components, reg=0.5)
                print(f"Extracted CSP features with high regularization, shape: {X_csp.shape}")
            except Exception as e:
                print(f"Error extracting CSP features with high regularization for subject {subject_id}: {str(e)}")
                X_csp, csp_transformer = None, None
        else:
            # Extract CSP features with robust method
            try:
                # Check for NaN/inf values before CSP extraction
                if np.isnan(X_raw).any() or np.isinf(X_raw).any():
                    print(f"Warning: Data for subject {subject_id} contains NaN/inf values. Attempting to clean...")
                    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
                    # Check again after cleaning
                    if np.isnan(X_raw).any() or np.isinf(X_raw).any():
                        print(f"Error: Could not clean NaN/inf values for subject {subject_id}. Skipping CSP.")
                        X_csp, csp_transformer = None, None
                    else:
                        X_csp, csp_transformer = extract_robust_csp_features(X_raw, y, n_components=n_csp_components, reg=0.1)
                        print(f"Extracted CSP features with shape: {X_csp.shape}")
                else:
                    X_csp, csp_transformer = extract_robust_csp_features(X_raw, y, n_components=n_csp_components, reg=0.1)
                    print(f"Extracted CSP features with shape: {X_csp.shape}")
            except Exception as e:
                print(f"Error extracting CSP features for subject {subject_id}: {str(e)}")
                # Try with increased regularization as fallback
                try:
                    print(f"Retrying with increased regularization...")
                    X_csp, csp_transformer = extract_robust_csp_features(X_raw, y, n_components=n_csp_components, reg=0.5)
                    print(f"Extracted CSP features with high regularization, shape: {X_csp.shape}")
                except Exception as e2:
                    print(f"Error extracting CSP features with high regularization: {str(e2)}")
                    X_csp, csp_transformer = None, None
        
        # Extract frequency features
        try:
            X_freq = extract_frequency_features(X_raw, bands=BANDS)
            print(f"Extracted frequency features with shape: {X_freq.shape}")
        except Exception as e:
            print(f"Error extracting frequency features for subject {subject_id}: {str(e)}")
            X_freq = None
        
        # Store data for this subject
        data[subject_id] = {
            'raw': X_raw,
            'labels': y,
            'csp': X_csp,
            'csp_transformer': csp_transformer,
            'freq': X_freq
        }
    
    return data


def cross_validate_model(data, subject_id, n_folds=5, n_epochs=100, 
                         output_dir=None, save_models=False, **model_params):
    """
    Perform cross-validation on a subject's data
    
    Args:
        data: Dictionary containing subject data
        subject_id: Subject ID
        n_folds: Number of folds for cross-validation
        n_epochs: Number of epochs for training
        output_dir: Directory to save output files
        save_models: Whether to save models for each fold
        model_params: Additional parameters for the model
        
    Returns:
        Dictionary with cross-validation results
    """
    print(f"Starting {n_folds}-fold cross-validation for subject {subject_id}")
    
    # Get subject data
    X_raw = data[subject_id]['raw']
    y = data[subject_id]['labels']
    X_csp = data[subject_id]['csp']
    X_freq = data[subject_id]['freq']
    
    # Set up cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Results storage
    results = {
        'val_acc': [],
        'val_balanced_acc': [],
        'conf_matrices': [],
        'training_time': [],
        'fold_indices': []
    }
    
    # Perform cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_raw)):
        print(f"\nFold {fold+1}/{n_folds}")
        
        # Create model
        n_classes = len(np.unique(y))
        model = HybridModelClassifier(
            n_classes=n_classes,
            n_epochs=n_epochs,
            **model_params
        )
        
        # Train and evaluate model
        fold_results = model.train_and_evaluate(
            X_raw, y, X_csp, X_freq,
            train_indices=train_idx, val_indices=val_idx
        )
        
        # Save results
        results['val_acc'].append(fold_results['val_acc'])
        results['val_balanced_acc'].append(fold_results['val_balanced_acc'])
        results['conf_matrices'].append(fold_results['conf_matrix'].tolist())
        results['training_time'].append(fold_results['training_time'])
        results['fold_indices'].append({
            'train': train_idx.tolist(),
            'val': val_idx.tolist()
        })
        
        # Save model if requested
        if save_models and output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            model_path = os.path.join(output_dir, f"model_S{subject_id}_fold{fold+1}.pt")
            model.save_model(model_path)
    
    # Calculate overall results
    results['mean_val_acc'] = np.mean(results['val_acc'])
    results['std_val_acc'] = np.std(results['val_acc'])
    results['mean_val_balanced_acc'] = np.mean(results['val_balanced_acc'])
    results['std_val_balanced_acc'] = np.std(results['val_balanced_acc'])
    results['mean_training_time'] = np.mean(results['training_time'])
    
    # Print summary
    print("\nCross-validation summary:")
    print(f"Mean validation accuracy: {results['mean_val_acc']:.4f} ± {results['std_val_acc']:.4f}")
    print(f"Mean validation balanced accuracy: {results['mean_val_balanced_acc']:.4f} ± {results['std_val_balanced_acc']:.4f}")
    print(f"Mean training time: {results['mean_training_time']:.2f} seconds")
    
    # Save results
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, f"cv_results_S{subject_id}.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_path}")
    
    return results


def pretrain_model(data, subject_ids, output_dir, n_epochs=200, **model_params):
    """
    Pretrain a model on data from multiple subjects
    
    Args:
        data: Dictionary containing data for all subjects
        subject_ids: List of subject IDs to use for pretraining
        output_dir: Directory to save the pretrained model
        n_epochs: Number of epochs for training
        model_params: Additional parameters for the model
        
    Returns:
        Pretrained model
    """
    print(f"Pretraining model on subjects: {subject_ids}")
    
    # Combine data from all subjects
    X_raw_all = []
    y_all = []
    X_csp_all = []
    X_freq_all = []
    
    # Track compatible subjects
    compatible_subjects = []
    reference_shape = None
    
    for i, subject_id in enumerate(subject_ids):
        if subject_id not in data:
            print(f"Data for subject {subject_id} not found. Skipping.")
            continue
        
        # Get raw data shape
        current_shape = data[subject_id]['raw'].shape[1:]  # Get (n_channels, n_times)
        
        # Set reference shape from first valid subject
        if reference_shape is None:
            reference_shape = current_shape
            print(f"Using reference data shape {reference_shape} from subject {subject_id}")
        
        # Check compatibility
        if current_shape != reference_shape:
            print(f"Skipping subject {subject_id}: Data shape {current_shape} doesn't match reference shape {reference_shape}")
            continue
        
        # Check for NaN values
        if np.isnan(data[subject_id]['raw']).any():
            print(f"Skipping subject {subject_id}: Contains NaN values in raw data")
            continue
            
        # Add data from compatible subject
        X_raw_all.append(data[subject_id]['raw'])
        y_all.append(data[subject_id]['labels'])
        
        if data[subject_id]['csp'] is not None:
            X_csp_all.append(data[subject_id]['csp'])
        
        if data[subject_id]['freq'] is not None:
            X_freq_all.append(data[subject_id]['freq'])
            
        # Track which subjects were compatible
        compatible_subjects.append(subject_id)
        
    if not X_raw_all:
        raise ValueError("No compatible subjects found for pretraining")
        
    print(f"Using {len(compatible_subjects)} compatible subjects for pretraining: {compatible_subjects}")
    
    # Concatenate data
    X_raw = np.concatenate(X_raw_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    
    # For CSP and frequency features, we need to recalculate on combined data
    n_classes = len(np.unique(y))
    
    # Extract CSP features from combined data
    try:
        X_csp, csp_transformer = extract_robust_csp_features(X_raw, y, n_components=model_params.get('n_csp_components', 4), reg=0.1)
    except Exception as e:
        print(f"Error extracting CSP features for combined data: {e}")
        # Try with higher regularization
        try:
            print("Retrying with higher regularization...")
            X_csp, csp_transformer = extract_robust_csp_features(X_raw, y, n_components=model_params.get('n_csp_components', 4), reg=0.5)
        except Exception as e2:
            print(f"Error extracting CSP features with higher regularization: {e2}")
            X_csp = None
    
    # Extract frequency features from combined data
    try:
        X_freq = extract_frequency_features(X_raw, bands=BANDS)
    except Exception as e:
        print(f"Error extracting frequency features for combined data: {e}")
        X_freq = None
    
    # Create and train model
    model = HybridModelClassifier(
        n_classes=n_classes,
        n_epochs=n_epochs,
        **model_params
    )
    
    # Train with more validation data to ensure generalization
    results = model.train_and_evaluate(
        X_raw, y, X_csp, X_freq,
        val_split=0.2  # Use 20% of data for validation
    )
    
    # Print results
    print("\nPretraining results:")
    print(f"Validation accuracy: {results['val_acc']:.4f}")
    print(f"Validation balanced accuracy: {results['val_balanced_acc']:.4f}")
    print(f"Training time: {results['training_time']:.2f} seconds")
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "pretrained_model.pt")
    model.save_model(model_path)
    print(f"Pretrained model saved to {model_path}")
    
    # Save CSP transformer
    if X_csp is not None:
        transformer_path = os.path.join(output_dir, "csp_transformer.npy")
        np.save(transformer_path, csp_transformer)
        print(f"CSP transformer saved to {transformer_path}")
    
    # Save compatible subjects list
    compatible_subjects_path = os.path.join(output_dir, "compatible_subjects.txt")
    with open(compatible_subjects_path, 'w') as f:
        for subject in compatible_subjects:
            f.write(f"{subject}\n")
    print(f"List of {len(compatible_subjects)} compatible subjects saved to {compatible_subjects_path}")
    
    return model


def finetune_subject_specific(base_model_path, data, subject_id, output_dir, 
                              n_epochs=100, n_csp_components=4):
    """
    Load a pretrained base model and fine-tune it for a specific subject.
    
    Args:
        base_model_path: Path to the pretrained model
        data: Dictionary containing subject data
        subject_id: Subject ID to fine-tune on
        output_dir: Directory to save the fine-tuned model
        n_epochs: Number of epochs for fine-tuning
        n_csp_components: Number of CSP components
        
    Returns:
        Dictionary of results
    """
    print(f"Fine-tuning for subject {subject_id}...")
    
    # Get subject data
    X_raw = data[subject_id]['raw']
    y = data[subject_id]['labels']
    X_csp = data[subject_id]['csp']
    X_freq = data[subject_id]['freq']
    
    n_channels, n_times = X_raw.shape[1], X_raw.shape[2]
    n_freq_features = X_freq.shape[1] if X_freq is not None else None
    n_classes = len(np.unique(y))
    
    # Define a custom training function with more aggressive scheduling
    def custom_train_evaluate_with_better_scheduling(X_raw, y, X_csp, X_freq, val_split):
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
        inv_freq = total_samples / (n_classes * class_counts)
        
        # Apply power function to make the weights more aggressive
        class_weights = np.power(inv_freq, weight_factor)
        
        # Normalize to keep average weight at n_classes
        class_weights = class_weights / np.mean(class_weights) * n_classes
        
        print(f"Using aggressive class weights: {class_weights}")
        
        # Create dataloaders with higher validation split for fine-tuning
        train_loader, val_loader, train_idx, val_idx = model._create_dataloaders(
            X_raw, y, X_csp, X_freq, val_split=val_split,
            apply_augmentation=True  # Use augmentation for fine-tuning
        )
        
        # Initialize with aggressive class weights
        model._initialize_training(class_weights)
        
        # Increase learning rate for fine-tuning to help escape local minima
        for param_group in model.optimizer.param_groups:
            param_group['lr'] = max(0.0003, model.lr * 2)  # Double the learning rate, with a floor
            param_group['weight_decay'] = 0.005  # Use lower weight decay for fine-tuning
        
        # Use a more aggressive scheduler
        model.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            model.optimizer, mode='max', factor=0.5, patience=10,
            min_lr=0.0001, verbose=True
        )
        
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
        patience = max(30, model.n_epochs // 5)  # Longer patience (20% of epochs)
        patience_counter = 0
        
        # Start training
        start_time = time.time()
        
        for epoch in range(model.n_epochs):
            # Training phase
            model.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # Iterate over training batches
            for batch in train_loader:
                model.optimizer.zero_grad()
                
                # Forward pass
                outputs, y_batch = model._forward_pass(batch)
                
                # Calculate loss
                loss = model.criterion(outputs, y_batch)
                
                # Backward pass and optimization
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=1.0)
                model.optimizer.step()
                
                # Update statistics
                train_loss += loss.item() * y_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == y_batch).sum().item()
                train_total += y_batch.size(0)
            
            # Calculate training metrics
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            
            # Validation phase with class balancing enabled
            # This helps prevent getting stuck in the "predict all one class" trap
            model.model.eval()
            val_loss = 0.0
            all_preds = []
            all_targets = []
            
            # Check if we should enable force_balanced during validation
            # Enable force_balanced after 1/3 of the way through training
            # or if the accuracy is suspiciously high compared to balanced accuracy
            enable_balanced = epoch > model.n_epochs // 3
            
            with torch.no_grad():
                for batch in val_loader:
                    # Forward pass with potential balancing
                    outputs, y_batch = model._forward_pass(batch, force_balanced=enable_balanced)
                    
                    # Calculate loss
                    loss = model.criterion(outputs, y_batch)
                    
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
            model.scheduler.step(val_balanced_acc)
            
            # Save history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['val_balanced_acc'].append(val_balanced_acc)
            
            # Print progress
            if (epoch + 1) % max(1, model.n_epochs // 20) == 0:
                print(f"Epoch {epoch+1}/{model.n_epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Train Acc: {train_acc:.4f} | "
                      f"Val Acc: {val_acc:.4f} | "
                      f"Val Balanced Acc: {val_balanced_acc:.4f}")
            
            # Check for improvement
            current_metric = val_balanced_acc  # Use balanced accuracy
            if current_metric > best_val_metric:
                best_val_metric = current_metric
                best_epoch = epoch
                patience_counter = 0
                
                # Save best model
                best_model_state = model.model.state_dict().copy()
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}. Best epoch: {best_epoch+1}")
                break
        
        # Load best model
        model.model.load_state_dict(best_model_state)
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Final evaluation
        all_preds = []
        all_targets = []
        
        model.model.eval()
        with torch.no_grad():
            for batch in val_loader:
                # Forward pass without balancing for final evaluation
                outputs, y_batch = model._forward_pass(batch)
                
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
            'model': model.model,
            'history': history,
            'best_epoch': best_epoch,
            'val_acc': final_val_acc,
            'val_balanced_acc': final_val_balanced_acc,
            'conf_matrix': conf_matrix,
            'training_time': training_time,
            'train_indices': train_idx,
            'val_indices': val_idx
        }
        
        print(f"Fine-tuning completed. Final validation accuracy: {final_val_acc:.4f}, balanced accuracy: {final_val_balanced_acc:.4f}")
        print(f"Training time: {training_time:.2f} seconds")
        print("Confusion matrix:")
        print(conf_matrix)
        
        # Save history for later use
        model.history = history
        
        return results
    
    # Create model for fine-tuning
    model = HybridModelClassifier(
        n_classes=n_classes,
        n_epochs=n_epochs,
        batch_size=16,  # Smaller batch size for fine-tuning
        weight_decay=0.005,  # Less regularization for fine-tuning
        augmentation_factor=3,  # More augmentation for fine-tuning
        force_balanced_val=True  # Enable balanced validation
    )
    
    # Load pretrained model
    model.load_model(
        base_model_path,
        n_channels=n_channels,
        n_times=n_times,
        n_freq_features=n_freq_features
    )
    
    # Fine-tune with custom training function
    results = model.train_and_evaluate(
        X_raw, y, X_csp, X_freq,
        val_split=0.4,  # Use more data for validation in fine-tuning
        custom_train_fn=custom_train_evaluate_with_better_scheduling
    )
    
    # Save fine-tuned model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"finetuned_model_S{subject_id}.pt")
    model.save_model(model_path)
    print(f"Fine-tuned model saved to {model_path}")
    
    # Save training curves
    if model.history is not None:
        fig_path = os.path.join(output_dir, f"training_curves_S{subject_id}.png")
        model.plot_training_curves(save_path=fig_path)
    
    return results


def evaluate_model_on_subjects(model_path, data, subject_ids, output_dir=None):
    """
    Evaluate a model on multiple subjects
    
    Args:
        model_path: Path to the model
        data: Dictionary containing subject data
        subject_ids: List of subject IDs to evaluate on
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary with evaluation results for each subject
    """
    print(f"Evaluating model on subjects: {subject_ids}")
    
    results = {}
    
    for subject_id in subject_ids:
        if subject_id not in data:
            print(f"Data for subject {subject_id} not found. Skipping.")
            continue
            
        print(f"\nEvaluating on subject {subject_id}")
        
        # Get subject data
        X_raw = data[subject_id]['raw']
        y = data[subject_id]['labels']
        X_csp = data[subject_id]['csp']
        X_freq = data[subject_id]['freq']
        
        n_channels, n_times = X_raw.shape[1], X_raw.shape[2]
        n_freq_features = X_freq.shape[1] if X_freq is not None else None
        n_classes = len(np.unique(y))
        
        # Create and load model
        model = HybridModelClassifier(n_classes=n_classes)
        model.load_model(
            model_path,
            n_channels=n_channels,
            n_times=n_times,
            n_freq_features=n_freq_features
        )
        
        # Evaluate model
        eval_results = model.evaluate(X_raw, y, X_csp, X_freq)
        
        # Store results
        results[subject_id] = {
            'accuracy': eval_results['accuracy'],
            'balanced_accuracy': eval_results['balanced_accuracy'],
            'accuracy_balanced': eval_results['accuracy_balanced'],
            'balanced_accuracy_balanced': eval_results['balanced_accuracy_balanced'],
            'confusion_matrix': eval_results['confusion_matrix'].tolist(),
            'confusion_matrix_balanced': eval_results['confusion_matrix_balanced'].tolist()
        }
        
    # Calculate average results
    if results:
        avg_results = {
            'accuracy': np.mean([r['accuracy'] for r in results.values()]),
            'balanced_accuracy': np.mean([r['balanced_accuracy'] for r in results.values()]),
            'accuracy_balanced': np.mean([r['accuracy_balanced'] for r in results.values()]),
            'balanced_accuracy_balanced': np.mean([r['balanced_accuracy_balanced'] for r in results.values()])
        }
        
        print("\nAverage results across subjects:")
        print(f"Accuracy: {avg_results['accuracy']:.4f}")
        print(f"Balanced Accuracy: {avg_results['balanced_accuracy']:.4f}")
        print(f"Accuracy (balanced): {avg_results['accuracy_balanced']:.4f}")
        print(f"Balanced Accuracy (balanced): {avg_results['balanced_accuracy_balanced']:.4f}")
        
        # Add average results
        results['average'] = avg_results
    
    # Save results
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, "evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Evaluation results saved to {results_path}")
    
    return results


def plot_confusion_matrices(results, output_dir=None):
    """
    Plot confusion matrices for each subject
    
    Args:
        results: Dictionary with evaluation results
        output_dir: Directory to save plots
    """
    # Skip average results
    subject_ids = [s for s in results.keys() if s != 'average']
    
    # Create figure
    n_subjects = len(subject_ids)
    fig, axs = plt.subplots(n_subjects, 2, figsize=(12, 5 * n_subjects))
    
    if n_subjects == 1:
        axs = np.array([axs])  # Make it indexable for single subject
    
    for i, subject_id in enumerate(subject_ids):
        # Get confusion matrices
        cm = np.array(results[subject_id]['confusion_matrix'])
        cm_balanced = np.array(results[subject_id]['confusion_matrix_balanced'])
        
        # Plot standard confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axs[i, 0])
        axs[i, 0].set_title(f"Subject {subject_id} - Standard")
        axs[i, 0].set_xlabel('Predicted')
        axs[i, 0].set_ylabel('True')
        
        # Plot balanced confusion matrix
        sns.heatmap(cm_balanced, annot=True, fmt='d', cmap='Blues', ax=axs[i, 1])
        axs[i, 1].set_title(f"Subject {subject_id} - Balanced")
        axs[i, 1].set_xlabel('Predicted')
        axs[i, 1].set_ylabel('True')
    
    plt.tight_layout()
    
    # Save plot
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "confusion_matrices.png")
        plt.savefig(plot_path)
        print(f"Confusion matrices saved to {plot_path}")
    
    return fig, axs 