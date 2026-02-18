import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import torch
import pandas as pd
from tqdm import tqdm  # Import tqdm for progress bars
import argparse
import mne
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold

from data_utils import load_multi_subject_data, load_subject_data
from data_augmentation import OptimizedAugmenter, SingleSubjectAugmenter, StrategyTester, BasicAugmenter, NoAugmenter
from hybrid_cnn_transformer import HybridModelClassifier
from visualization import plot_base_model_results, plot_strategy_results, plot_strategy_comparison 
from visualization import visualize_cross_subject_results, average_training_curves, single_training_curve
from feature_modules import CSPModule, SpectrogramModule
import classical_baseline

# Global configuration
SUBJECT_ID = "001"           # Single subject to focus on
RUN_IDS = ["3", "7", "11"]   # Motor imagery runs
RANDOM_SEED = 2             # For reproducibility
USE_MOTOR_CHANNELS = True    # Whether to select only motor-related channels
DEBUG = False                # Global debug flag to control printouts
FILTER_BANK_CSP = False      # Flag to enable filter-bank CSP features

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Set device for PyTorch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Debug print function
def dprint(*args, **kwargs):
    """Only print if DEBUG is True"""
    if DEBUG:
        print(*args, **kwargs)

def train_base_model_multi_subject(run_ids=RUN_IDS, n_subjects=50, 
                                   save_model=True, n_epochs=200, 
                                   model_name="base_model", subject_ids=None,
                                   val_subject_ids=None,
                                   model_spec={}, rng_seed=RANDOM_SEED,
                                   augmenter=None,
                                   feature_modules=[]):
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
        val_subject_ids: Specific subject IDs to use for validation (if None, random 20% hold-out)
        model_type: Which backbone to use ('transformer' or 'mamba').
        
    Returns:
        Trained classifier and evaluation results
    """
    dprint(f"Training base model on up to {n_subjects} subjects...")
    
    # Load multi-subject data with specific subject IDs if provided
    X, y, subject_indices = load_multi_subject_data(
        subject_ids=subject_ids, 
        run_ids=run_ids,
        use_motor_channels=USE_MOTOR_CHANNELS,
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
    
    n_subjects_actual = len(unique_subject_indices)
    dprint(f"\nTraining base model on {n_subjects_actual} subjects")
    dprint(f"Subjects included: {used_subject_ids}")
    dprint(f"Data shape: {X.shape} (trials, channels, time points)")
    dprint(f"Number of classes: {len(np.unique(y))}")
    dprint(f"Class distribution: {np.bincount(y)}")
    
    # Create optimized augmenter (will apply only to the training split if none provided)
    if augmenter is None:
        augmenter = OptimizedAugmenter(augmentation_factor=model_spec['augmentation_factor'])
    
    # Create classifier with optimized hyperparameters for multi-subject learning
    n_classes = len(np.unique(y))
    classifier = HybridModelClassifier(
        n_classes=n_classes,
        embedding_dim=model_spec['embedding_dim'],
        n_heads=model_spec['n_heads'],
        n_layers=model_spec['n_layers'],
        dropout=model_spec['dropout'],
        lr=model_spec['lr'],
        batch_size=model_spec['batch_size'],
        n_epochs=n_epochs,
        weight_decay=model_spec['weight_decay'],
        use_feature_modules=True,
        model_type=model_spec['model_type']
    )
    
    # 1) Split raw trials by subject, allowing explicit validation folds
    unique_subjects = np.unique(subject_indices)
    if val_subject_ids is None:
        # Random 20% hold-out if no explicit validation subjects provided
        rng = np.random.RandomState(rng_seed)
        rng.shuffle(unique_subjects)
        n_val = max(1, int(0.2 * len(unique_subjects)))
        val_subjects = unique_subjects[:n_val]
        train_subjects = unique_subjects[n_val:]
    else:
        # Use explicit validation subjects by ID
        val_subjects = []
        for vsid in val_subject_ids:
            if subject_ids is not None and vsid in subject_ids:
                val_subjects.append(subject_ids.index(vsid))
            else:
                raise ValueError(f"Validation subject {vsid} not found in provided subject_ids")
        train_subjects = [idx for idx in unique_subjects if idx not in val_subjects]
    assert set(train_subjects).isdisjoint(val_subjects)

    train_mask = np.isin(subject_indices, train_subjects)
    val_mask   = np.isin(subject_indices, val_subjects)

    X = torch.FloatTensor(X).to(DEVICE)
    y = torch.LongTensor(y).to(DEVICE)
    subject_indices = torch.LongTensor(subject_indices).to(DEVICE)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]

    # Prepare subject indices for train and validation splits
    train_subject_indices = subject_indices[train_mask]
    val_subject_indices = subject_indices[val_mask]

    # 2) Augment only the train set (CSP fitted on unaugmented data)
    X_train_aug, y_train_aug = augmenter.augment_batch(X_train, y_train)

    if 'csp_bands' in model_spec:
        filter_bank = True
        bands = model_spec['csp_bands']
        classifier.n_csp_components = len(bands) * model_spec['n_csp_components']
    else:
        filter_bank = False
        bands = None

    # Split each validation subject's trials: 75% for feature fit, 25% held-out for evaluation
    val_global_idxs = np.where(val_mask)[0]
    # Stratified split per validation subject to hold out 25% for evaluation
    val_fit_mask = np.zeros_like(val_mask)
    val_eval_mask = np.zeros_like(val_mask)
    # Convert labels to numpy for splitting
    y_np = y.cpu().numpy()
    subj_idx_np = subject_indices.cpu().numpy()
    for subj in np.unique(subj_idx_np[val_global_idxs]):
        subj_trials = val_global_idxs[subj_idx_np[val_global_idxs] == subj]
        subj_labels = y_np[subj_trials]
        # train_test_split returns X_train, X_test, y_train, y_test
        fit_idxs, eval_idxs, _, _ = train_test_split(
            subj_trials, subj_labels,
            test_size=0.25,
            stratify=subj_labels,
            random_state=RANDOM_SEED
        )
        val_fit_mask[fit_idxs] = True
        val_eval_mask[eval_idxs] = True

    # Combine training and val-fit trials for module fitting (unaugmented)
    X_fm = torch.cat([X_train, X[val_fit_mask]], dim=0)
    y_fm = torch.cat([y_train, y[val_fit_mask]], dim=0)
    subj_fm = torch.cat([train_subject_indices, subject_indices[val_fit_mask]], dim=0)
    # Fit feature modules on the combined fit set
    for mod in feature_modules:
        mod.fit(X_fm, y_fm, subject_indices=subj_fm)
    classifier.feature_modules = feature_modules

    # Prepare held-out validation data
    X_val_eval = X[val_eval_mask]
    y_val_eval = y[val_eval_mask]
    val_eval_subject_indices = subject_indices[val_eval_mask]

    # 3) Train, letting classifier handle feature transforms internally on held-out trials
    results = classifier.train_and_evaluate(
        X_train_aug, y_train_aug,
        X_val=X_val_eval, y_val=y_val_eval,
        train_subject_indices=train_subject_indices,
        val_subject_indices=val_eval_subject_indices,
        early_stopping_patience=40,
        use_lr_scheduler='onecycle'
    )
    
    # Add the trained classifier to the results
    results['classifier'] = classifier
    
    # Create and save plots using the visualization module
    plot_base_model_results(results, n_subjects_actual)
    
    # Record which subjects were held-out for validation
    metadata = {
        'training_subjects': used_subject_ids,
        'validation_subjects': (
            val_subject_ids 
            if val_subject_ids is not None 
            else [
                subject_ids[i] if subject_ids is not None else f"{int(i)+1:03d}"
                for i in val_subjects
            ]
        ),
        'n_subjects': n_subjects_actual,
        'requested_n_subjects': n_subjects,  # Store the requested number
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'run_ids': run_ids
    }
    
    # Save the model with metadata
    if save_model:
        model_dir = os.path.join(os.getcwd(), 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        # Use the *actual* number of subjects in the filename to avoid
        # misleading checkpoints when fewer subjects are available locally.
        model_path = os.path.join(model_dir, f'{model_name}_{n_subjects_actual}_subjects.pt')
        classifier.save_model(model_path, metadata=metadata)
        
        # Also save subjects list to a separate text file for easy reference
        subjects_path = os.path.join(model_dir, f'{model_name}_{n_subjects_actual}_subjects_list.txt')
        with open(subjects_path, 'w') as f:
            f.write(f"# Training subjects for model: {model_name}_{n_subjects_actual}_subjects.pt\n")
            f.write(f"# Trained on: {metadata['timestamp']}\n")
            f.write(f"# Requested subjects: {n_subjects}\n")
            f.write(f"# Actual subjects used: {n_subjects_actual}\n\n")
            for subject_id in used_subject_ids:
                f.write(f"{subject_id}\n")
        
        dprint(f"Base model saved to {model_path}")
        dprint(f"Subject list saved to {subjects_path}")
        
        # Print a warning if the actual number differs from the requested number
        if n_subjects_actual != n_subjects:
            dprint(f"\nWARNING: {n_subjects - n_subjects_actual} subjects were excluded during data loading.")
            dprint(f"Requested: {n_subjects} subjects, Actual: {n_subjects_actual} subjects")
            dprint("This is likely because some subjects had non-standard time dimensions.")
            dprint(f"The model is saved using the actual number: {model_name}_{n_subjects_actual}_subjects.pt")
    
    return classifier, results, metadata


def augment_middle_percentage(X, y, augmenter, percentage=50):
    """
    Apply augmentation to the middle X% of trials to avoid augmenting outliers.
    
    Args:
        X: Input data array of shape (n_samples, n_channels, n_times)
        y: Labels array of shape (n_samples,)
        augmenter: Augmentation object with augment_batch method
        percentage: Percentage of data to augment (default: 50%)
        
    Returns:
        X_augmented: Combined original and augmented data
        y_augmented: Combined original and augmented labels
    """
    # If no augmenter provided, return original data
    if augmenter is None:
        dprint("No augmenter provided - returning original data without augmentation")
        return X, y
    
    # Sort indices by energy/variance to identify potential outliers
    trial_energy = torch.var(X, dim=(1, 2))
    sorted_indices = torch.argsort(trial_energy)
    
    # Get middle X% indices
    n_samples = len(sorted_indices)
    start_idx = int(n_samples * ((100 - percentage) / 200))
    end_idx = int(n_samples * (1 - (100 - percentage) / 200))
    middle_indices = sorted_indices[start_idx:end_idx]
    
    dprint(f"Augmenting the middle {percentage}% ({len(middle_indices)}/{n_samples}) trials")
    
    # Extract middle X% data for augmentation
    X_to_augment = X[middle_indices]
    y_to_augment = y[middle_indices]
    
    # Apply augmentation to the middle X%
    dprint(f"Before augmentation: {len(X_to_augment)} samples")
    X_augmented_middle, y_augmented_middle = augmenter.augment_batch(X_to_augment, y_to_augment)
    dprint(f"After augmentation: {len(X_augmented_middle)} samples")
    dprint(f"Augmentation multiplier: {len(X_augmented_middle) / len(X_to_augment):.2f}x")
    
    # Combine original and augmented data (handle numpy arrays or torch tensors)
    X_combined = torch.cat([X, X_augmented_middle], dim=0)
    y_combined = torch.cat([y, y_augmented_middle], dim=0)
    
    return X_combined, y_combined


def finetune_subject_specific(base_model_path, subject_id, run_ids=RUN_IDS, 
                             n_epochs=200, save_models=True, val_split=0.2, freeze_layers=True,
                             use_cross_validation=True, n_folds=5, augmentation_factor=10,
                             early_stopping_patience=15, augment_percentage=50,
                             track_test_acc=True):
    """
    Load a pretrained base model and fine-tune it for a specific subject.
    Tests different augmentation strategies using cross-validation.
    
    Args:
        base_model_path: Path to the pretrained model
        subject_id: Subject ID to fine-tune on
        run_ids: Run IDs to use
        n_epochs: Number of epochs for fine-tuning
        save_models: Whether to save the fine-tuned models
        val_split: Proportion of data to use for validation (used only if use_cross_validation=False)
        freeze_layers: Whether to freeze all layers except the classifier
        use_cross_validation: Whether to use k-fold cross-validation
        n_folds: Number of folds for cross-validation
        augmentation_factor: Factor by which to augment data (default: 10)
        early_stopping_patience: Non-improving epochs for early stopping (default: 15)
        augment_percentage: Percentage of middle data to augment (default: 50)
        track_test_acc: Whether to track test accuracy during training (default: True)
    Returns:
        Dictionary of results for different augmentation strategies
    """
    dprint(f"Fine-tuning for subject {subject_id}...")
    
    # Load subject data
    X, y, info = load_subject_data(subject_id=subject_id, run_ids=run_ids, use_motor_channels=USE_MOTOR_CHANNELS)
    
    # Print dataset information
    dprint(f"Data shape: {X.shape} (trials, channels, time points)")
    dprint(f"Number of classes: {len(np.unique(y))}")
    dprint(f"Class distribution: {np.bincount(y)}")
    
    # First split data into 80% for training/validation and 20% for final testing
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED, shuffle=True
    )
    
    dprint(f"Split data: {X_trainval.shape[0]} train+val samples, {X_test.shape[0]} test samples")
    dprint(f"Train+val class distribution: {np.bincount(y_trainval)}")
    dprint(f"Test class distribution: {np.bincount(y_test)}")

    # Create strategy tester to get augmenters
    strategy_tester = StrategyTester(augmentation_factor=augmentation_factor)
    
    # Debug augmentation factor
    dprint(f"Using augmentation_factor={augmentation_factor}")
    
    # Define augmentation strategies to test for fine-tuning
    aug_strategies = [
        {
            'name': 'None', 
            'augmenter': NoAugmenter()
        },
        {
            'name': 'All Combined',
            'augmenter': SingleSubjectAugmenter(augmentation_factor=augmentation_factor)
        }
    ]
    
    # Add individual strategy augmenters
    individual_strategies = [
        'add_noise', 
        'scale_amplitude', 
        'spectral_perturbation',
        'smooth_warping',
        'frequency_band_noise',
        'channel_dropout',
        'temporal_shift',
        'neural_jitter',
        'trial_mixup'
    ]
    
    for strategy in individual_strategies:
        aug_strategies.append({
            'name': f'Only {strategy}',
            'augmenter': strategy_tester.get_single_strategy_augmenter(strategy)
        })
    
    # Add combined strategies that might work well
    effective_combinations = [
        {
            'name': 'Temporal Combo',
            'strategies': ['temporal_shift', 'neural_jitter', 'smooth_warping'],
            'weights': [0.3, 0.4, 0.3]
        },
        {
            'name': 'Spectral Combo',
            'strategies': ['spectral_perturbation', 'frequency_band_noise'],
            'weights': [0.5, 0.5]
        },
        {
            'name': 'Noise Combo',
            'strategies': ['add_noise', 'channel_dropout'],
            'weights': [0.6, 0.4]
        },
        {
            'name': 'Best Combo',
            'strategies': ['spectral_perturbation', 'neural_jitter', 'trial_mixup', 'smooth_warping'],
            'weights': [0.3, 0.2, 0.3, 0.2]
        }
    ]
    
    for combo in effective_combinations:
        aug_strategies.append({
            'name': combo['name'],
            'augmenter': strategy_tester.get_combined_strategy_augmenter(
                combo['strategies'], 
                combo['weights']
            )
        })
    
    results = {
        'strategies': [s['name'] for s in aug_strategies],
        'accuracy': [],
        'test_accuracy': [],  # Added test accuracy
        'fold_accuracies': [],
        'histories': [],
        'confusion_matrices': [],
        'test_confusion_matrices': []  # Added test confusion matrices
    }
    
    # Process each strategy with a progress bar if DEBUG is on, otherwise iterate without progress bar
    if DEBUG:
        strategy_iter = tqdm(aug_strategies, desc=f"Strategies for subject {subject_id}", position=0)
    else:
        strategy_iter = aug_strategies

    X_trainval = torch.FloatTensor(X_trainval).to(DEVICE)
    y_trainval = torch.LongTensor(y_trainval).to(DEVICE)
    X_test = torch.FloatTensor(X_test).to(DEVICE)
    y_test = torch.LongTensor(y_test).to(DEVICE)
    
    for i, strategy in enumerate(strategy_iter):
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        # Update description if using progress bar
        if DEBUG:
            strategy_iter.set_description(f"Strategy: {strategy['name']}")
            
        dprint(f"\n{'='*50}")
        dprint(f"Fine-tuning with {strategy['name']} augmentation")
        dprint(f"{'='*50}")
        
        # Set augmentation parameters
        augmenter = strategy['augmenter']
        
        # Print strategy information
        dprint(f"Using {strategy['name']} augmentation strategy")

        # Create folds with ORIGINAL (non-augmented) data
        if use_cross_validation:
            dprint(f"Using {n_folds}-fold cross-validation for evaluation")
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
            folds = list(skf.split(X_trainval.cpu(), y_trainval.cpu()))
        else:
            # For non-CV case, create a single "fold" with a train/val split
            dprint(f"Using a single train/val split with {val_split*100:.0f}% validation")
            X_train, X_val, y_train, y_val = train_test_split(
                X_trainval, y_trainval, test_size=val_split, stratify=y_trainval, random_state=RANDOM_SEED
            )
            # Create a single fold with indices
            train_idx = np.arange(len(X_train))
            val_idx = np.arange(len(X_train), len(X_train) + len(X_val))
            # Combine X_train and X_val back for consistent fold extraction
            X_trainval_combined = np.concatenate([X_train, X_val], axis=0)
            y_trainval_combined = np.concatenate([y_train, y_val], axis=0)
            folds = [(train_idx, val_idx)]
            # Replace trainval data with the combined version for this case
            X_trainval = X_trainval_combined
            y_trainval = y_trainval_combined

        # Enhanced fine-tuning parameters
        batch_size = 128
        initial_lr = 1e-4
        weight_decay = 0.001
        dropout = 0.05
        
        # Set up variables to store results
        fold_histories = []
        fold_accuracies = []
        fold_test_accuracies = []
        fold_confusion_matrices = []
        fold_test_confusion_matrices = []
        
        # Fold iteration (either k-fold CV or single train/val split)
        if DEBUG:
            fold_iter = tqdm(enumerate(folds), total=len(folds), desc="Folds", position=1, leave=False)
        else:
            fold_iter = enumerate(folds)
        
        for fold_idx, (train_idx, val_idx) in fold_iter:
            # Update description if using progress bar
            if DEBUG:
                fold_iter.set_description(f"Fold {fold_idx+1}/{len(folds)}")
            
            # Split data for this fold
            dprint(f"Train indices: {train_idx}, Val indices: {val_idx}")
            X_train_fold, X_val_fold = X_trainval[train_idx], X_trainval[val_idx]
            y_train_fold, y_val_fold = y_trainval[train_idx], y_trainval[val_idx]
            
            dprint(f"Train samples: {len(X_train_fold)}, Validation samples: {len(X_val_fold)}")
            dprint(f"Train class distribution: {torch.bincount(y_train_fold)}")
            dprint(f"Validation class distribution: {torch.bincount(y_val_fold)}")
            
            # Now augment both training and validation data separately
            if augmenter is not None:
                dprint(f"Augmenting training data with {strategy['name']}")
                X_train_aug, y_train_aug = augment_middle_percentage(
                    X_train_fold, y_train_fold, augmenter, percentage=augment_percentage
                )
                
                dprint(f"Augmenting validation data with {strategy['name']}")
                X_val_aug, y_val_aug = augment_middle_percentage(
                    X_val_fold, y_val_fold, augmenter, percentage=100
                )
                
                dprint(f"Original train samples: {len(X_train_fold)}, after augmentation: {len(X_train_aug)}")
                dprint(f"Original val samples: {len(X_val_fold)}, after augmentation: {len(X_val_aug)}")
            else:
                # No augmentation case
                X_train_aug, y_train_aug = X_train_fold, y_train_fold
                X_val_aug, y_val_aug = X_val_fold, y_val_fold
            
            # Load a fresh model for each fold
            base_classifier, metadata = HybridModelClassifier.load_model(base_model_path)

            # Build feature modules from base model metadata
            feature_module_configs = metadata.get('feature_modules')
            if feature_module_configs is None:
                raise ValueError("No feature module configuration found in base model metadata.")
            modules = []
            # Create dummy subject indices for CSP fitting
            subject_indices_train = torch.zeros(len(y_train_aug), dtype=int)
            subject_indices_val   = torch.zeros(len(y_val_aug),   dtype=int)
            for cfg in feature_module_configs:
                name = cfg['name']
                params = cfg['params']
                if name == 'CSPModule':
                    mod = CSPModule(**params)
                    mod.fit(X_train_aug, y_train_aug, subject_indices=subject_indices_train)
                    modules.append(mod)
                elif name == 'SpectrogramModule':
                    mod = SpectrogramModule(**params)
                    modules.append(mod)
                else:
                    raise ValueError(f"Unsupported feature module type: {name}")
            # Assign feature modules to classifier
            base_classifier.feature_modules = modules
            
            # Update classifier parameters
            base_classifier.n_epochs = n_epochs
            base_classifier.lr = initial_lr
            base_classifier.batch_size = batch_size
            base_classifier.weight_decay = weight_decay
            base_classifier.dropout = dropout
            
            # Freeze layers if requested
            if freeze_layers:
                # Only unfreeze adapter_head and combiner_head during fine tuning to prevent overfitting
                # unfrozen_layers = ['adapter_head', 'combiner_head']
                unfrozen_layers = [
                    'fusion',          # the scalar α (and small post-Linear)
                    'feature_encoder',     # lets the CSP vector re-centre to the new subject
                    'adapter_head',    # your light B-branch
                    'combiner_head'    # mixes A & B
                ]
                dprint(f"Freezing all layers except {unfrozen_layers}")
                base_classifier.freeze_layers(unfrozen_layers=unfrozen_layers)
            
            lr_scheduler = 'onecycle'
            # Fine-tune on this fold's training data
            if track_test_acc:
                subject_indices_test = torch.zeros(len(y_test), dtype=int)
                fold_results = base_classifier.train_and_evaluate(
                    X_train_aug, y_train_aug, X_val_aug, y_val_aug,  # Pass augmented validation set
                    train_subject_indices=subject_indices_train,
                    val_subject_indices=subject_indices_val,
                    early_stopping_patience=early_stopping_patience, 
                    fine_tuning=True,
                    use_lr_scheduler=lr_scheduler,
                    X_test=X_test, y_test=y_test,  # Add test data to track accuracy during training
                    test_subject_indices=subject_indices_test
                )
            else:
                fold_results = base_classifier.train_and_evaluate(
                    X_train_aug, y_train_aug, X_val_aug, y_val_aug,  # Pass augmented validation set
                    train_subject_indices=subject_indices_train,
                    val_subject_indices=subject_indices_val,
                    early_stopping_patience=early_stopping_patience, 
                    fine_tuning=True,
                    use_lr_scheduler=lr_scheduler
                )
            
            
            # Evaluate on the test set (features are now extracted internally)
            test_results = base_classifier.evaluate(X_test, y_test, subject_indices=subject_indices_test)
            
            # Plot and save the accuracy curves
            # if 'test_acc' in fold_results['history']:
            #     single_training_curve(fold_results, subject_id, strategy['name'], fold_idx)
            
            # Store results for this fold
            fold_accuracies.append(fold_results['accuracy'])
            fold_test_accuracies.append(test_results['accuracy'])
            fold_histories.append(fold_results['history'])
            fold_confusion_matrices.append(fold_results['confusion_matrix'])
            fold_test_confusion_matrices.append(test_results['confusion_matrix'])
            
            # Update progress bar postfix if in DEBUG mode
            if DEBUG:
                fold_iter.set_postfix(val_acc=f"{fold_results['accuracy']:.4f}", test_acc=f"{test_results['accuracy']:.4f}")
            
            # Save the model if requested (for non-CV case or if saving all fold models)
            if save_models and (not use_cross_validation or fold_idx == np.argmax(fold_accuracies)):
                model_dir = os.path.join(os.getcwd(), 'models')
                os.makedirs(model_dir, exist_ok=True)
                fold_suffix = f"_fold{fold_idx+1}" if use_cross_validation else ""
                model_path = os.path.join(model_dir, 
                                        f'finetuned/finetuned_subject_{subject_id}_{strategy["name"].replace(" ", "_")}{fold_suffix}.pt')
                base_classifier.save_model(model_path)
                dprint(f"Fine-tuned model saved to {model_path}")
        
        # Calculate average performance across folds (or just the single fold's performance)
        mean_accuracy = np.mean(fold_accuracies)
        std_accuracy = np.std(fold_accuracies)
        mean_test_accuracy = np.mean(fold_test_accuracies)
        std_test_accuracy = np.std(fold_test_accuracies)
        
        # Update strategy progress bar postfix if in DEBUG mode
        if DEBUG:
            strategy_iter.set_postfix(val_acc=f"{mean_accuracy:.4f}", test_acc=f"{mean_test_accuracy:.4f}")

        dprint(f"  {strategy['name']} - Mean Val: {mean_accuracy:.4f} ± {std_accuracy:.4f}, Mean Test: {mean_test_accuracy:.4f} ± {std_test_accuracy:.4f}")
        
        # Print fold summary
        if use_cross_validation:
            dprint(f"\nCross-validation result for {strategy['name']}:")
            dprint(f"Mean validation accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
            dprint(f"Mean test accuracy: {mean_test_accuracy:.4f} ± {std_test_accuracy:.4f}")
            dprint(f"Individual fold validation accuracies: {fold_accuracies}")
            dprint(f"Individual fold test accuracies: {fold_test_accuracies}")
        else:
            dprint(f"\nSingle-split result for {strategy['name']}:")
            dprint(f"Validation accuracy: {mean_accuracy:.4f}")
            dprint(f"Test accuracy: {mean_test_accuracy:.4f}")
        
        # Store overall results
        results['accuracy'].append(mean_accuracy)
        results['test_accuracy'].append(mean_test_accuracy)
        results['fold_accuracies'].append(fold_accuracies)
        results['histories'].append(fold_histories)
        results['confusion_matrices'].append(fold_confusion_matrices)
        results['test_confusion_matrices'].append(fold_test_confusion_matrices)
        
        # Plot average accuracy curves across all folds
        if use_cross_validation and len(fold_histories) > 1:
            average_training_curves(fold_histories, subject_id, strategy['name'])
        
        # Use visualization module to plot results for each strategy
        # Only create plot for the single run case as CV plots are handled later
        if not use_cross_validation:
            plot_strategy_results(fold_results, test_results, strategy['name'])
    
    # Print summary of results
    sorted_indices = np.argsort(results['accuracy'])[::-1]  # Descending order
    sorted_strategies = [results['strategies'][i] for i in sorted_indices]
    sorted_accuracies = [results['accuracy'][i] for i in sorted_indices]
    sorted_test_accuracies = [results['test_accuracy'][i] for i in sorted_indices]
    
    dprint("\nFine-tuning Results Summary:")
    dprint(f"{'Strategy':<25} {'Val Accuracy':<15} {'Test Accuracy':<15}")
    dprint('-' * 60)
    
    for strategy, accuracy, test_accuracy in zip(sorted_strategies, sorted_accuracies, sorted_test_accuracies):
        if use_cross_validation:
            fold_accs = results['fold_accuracies'][results['strategies'].index(strategy)]
            dprint(f"{strategy:<25} {accuracy:.4f} ± {np.std(fold_accs):.4f}    {test_accuracy:.4f}")
        else:
            dprint(f"{strategy:<25} {accuracy:.4f}    {test_accuracy:.4f}")
    
    # Find the best strategy
    best_idx = np.argmax(results['accuracy'])
    best_strategy = results['strategies'][best_idx]
    if use_cross_validation:
        best_folds = results['fold_accuracies'][best_idx]
        dprint(f"\nBest augmentation strategy for Subject {subject_id}: {best_strategy} "
              f"(Mean Val Accuracy: {results['accuracy'][best_idx]:.4f} ± {np.std(best_folds):.4f}, "
              f"Test Accuracy: {results['test_accuracy'][best_idx]:.4f})")
    else:
        dprint(f"\nBest augmentation strategy for Subject {subject_id}: {best_strategy} "
              f"(Val Accuracy: {results['accuracy'][best_idx]:.4f}, "
              f"Test Accuracy: {results['test_accuracy'][best_idx]:.4f})")
    
    # Create and save strategy comparison plot
    plot_strategy_comparison(results, subject_id, save_models)
    
    return results


def split_subjects(total_subjects=109, test_subjects_count=10, random_seed=RANDOM_SEED):
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
    
    dprint(f"Split {len(all_subjects)} subjects into {len(train_subjects)} for training and {len(test_subjects)} for testing")
    return train_subjects, test_subjects

def base_model_parameter_sweep(model_spec: dict):
    """
    Perform a parameter sweep for the base model.
    
    Args:
        model_spec: Dictionary containing model configuration
    """

    combinations = [{}]

    for combination in combinations:
        accs = []
        for fold_seed in range(3):
            for key, value in combination.items():
                model_spec[key] = value

            _, base_results, _ = train_base_model_multi_subject(
                run_ids=RUN_IDS,
                n_subjects=subjects,      # Use specified number of subjects for pretraining
                subject_ids=train_subjects,  # Only use training subjects
                save_model=False,
                n_epochs=300, 
                model_name=base_model_name,
                model_spec=model_spec,
                rng_seed=fold_seed
            )
            accs.append(base_results['accuracy'])

        for key, value in combination.items():
            print(f"{key}: {value}", end=", ")
        print(f"{np.mean(accs)}, +/- {np.std(accs)}, {list(accs)}")
    


if __name__ == "__main__":
    import time
    import os.path
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='EEG Classification Training Script')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--pretrain', '-pt', action='store_true', help='Run pretraining')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing models')
    parser.add_argument('--finetune', '-ft', action='store_true', help='Run finetuning')
    parser.add_argument('--ablation', '-ab', action='store_true', help='Run ablation study')
    parser.add_argument('--mamba', action='store_true', help='Use the Mamba backbone instead of the transformer')
    parser.add_argument('--basesweep', action='store_true', help='Run base model parameter sweep')
    parser.add_argument('--baseline', '-bl', action='store_true', help='Evaluate baseline accuracy (no fine-tuning) on test subjects')
    parser.add_argument('--classical', '-cl', action='store_true', help='Evaluate classical Riemannian tangent-space')
    parser.add_argument('--aug-sweep', action='store_true', help='Run augmentation strategy sweep')
    args = parser.parse_args()
    
    # Set debug flag globally
    DEBUG = args.debug
    
    # Update the module's DEBUG flag
    sys.modules['hybrid_cnn_transformer'].DEBUG = DEBUG
    sys.modules['data_utils'].DEBUG = DEBUG
    sys.modules['feature_utils'].DEBUG = DEBUG
    sys.modules['data_augmentation'].DEBUG = DEBUG
    sys.modules['visualization'].DEBUG = DEBUG

    sys.modules['hybrid_cnn_transformer'].RANDOM_SEED = RANDOM_SEED
    
    # Also set MNE verbosity level
    mne.set_log_level('INFO' if DEBUG else 'WARNING')
    
    print(f"Debug mode: {'ON' if DEBUG else 'OFF'}")
    
    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    
    test_subjects = 10
    subjects = 90

    # Step 0: Split subjects into training and test sets
    # Reserve 10 subjects for testing/fine-tuning
    train_subjects, test_subjects = split_subjects(total_subjects=109, test_subjects_count=test_subjects, random_seed=RANDOM_SEED)
    
    # Print subject split information
    dprint("Training subjects:", train_subjects[:5], "...", f"(total: {len(train_subjects)})")
    dprint("Test subjects:", test_subjects)
    
    # Define the model path and choose name based on backbone
    model_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(model_dir, exist_ok=True)
    # Append `_mamba` if using Mamba backbone
    if args.mamba:
        print("Using Mamba backbone")
        base_model_name = "base_model_mamba"
        model_spec = {
            'model_type': 'mamba',
            'embedding_dim': 256,
            'n_heads': 2,
            'n_layers': 1,
            'dropout': 0.4,
            'n_csp_components': 14,
            'batch_size': 512,
            'lr': 0.001,
            'weight_decay': 1e-4,
            'augmentation_factor': 4,
            'csp_bands': [(8, 10), (10, 12), (12, 15), (15, 18), (18, 21), (21, 24), (24, 27), (27, 30)],
        }
    else:
        print("Using Transformer backbone")
        base_model_name = "base_model"
        model_spec = {
            'model_type': 'transformer',
            'embedding_dim': 16,
            'n_heads': 2,
            'n_layers': 3,
            'dropout': 0.1,
            'n_csp_components': 6,
            'batch_size': 512,
            'lr': 1e-4,
            'weight_decay': 1e-5,
            'augmentation_factor': 4,
            'csp_bands': [(8, 10), (10, 12), (12, 15), (15, 18), (18, 21), (21, 24), (24, 27), (27, 30)],
        }

    # Set filter-bank CSP flag globally
    FILTER_BANK_CSP = 'csp_bands' in model_spec

    # Determine number of channels for spectrogram features
    # Sample first training subject to infer channel count
    first_subj = train_subjects[0]
    X_temp, _, _ = load_subject_data(subject_id=first_subj, run_ids=RUN_IDS, use_motor_channels=USE_MOTOR_CHANNELS)
    n_channels = X_temp.shape[1]

    # Set up feature modules for plug-and-play
    feature_modules = [
        CSPModule(
            n_components=model_spec['n_csp_components'],
            per_subject=True,
            filter_bank=FILTER_BANK_CSP,
            bands=model_spec['csp_bands']
        ),
        # SpectrogramModule(
        #     n_channels=n_channels,
        #     n_fft=64,
        #     hop_length=None,
        #     hidden=32,
        #     out_dim=model_spec['embedding_dim'],
        #     dropout=model_spec['dropout']
        # )
    ]
    n_subjects = subjects
    
    # Look for existing model files and train base model if needed
    potential_model_path = os.path.join(model_dir, f'{base_model_name}_{n_subjects}_subjects.pt')  
    if args.pretrain and (args.overwrite or not os.path.exists(potential_model_path)):
        print("Training new base model...")

        base_classifier, base_results, metadata = train_base_model_multi_subject(
            run_ids=RUN_IDS,
            n_subjects=subjects,      # Use specified number of subjects for pretraining
            subject_ids=train_subjects,  # Only use training subjects
            save_model=True,
            n_epochs=300, 
            model_name=base_model_name,
            model_spec=model_spec,
            feature_modules=feature_modules
        )
        base_model_path = os.path.join(model_dir, f'{base_model_name}_{metadata["n_subjects"]}_subjects.pt')
    elif args.basesweep:
        base_model_parameter_sweep(model_spec)
    elif os.path.exists(potential_model_path):
        base_model_path = potential_model_path
        print(f"Using existing base model: {base_model_path}")
    elif not args.classical and not args.aug_sweep:
        raise ValueError(f"Model not set to pretrain but no existing base model found at {potential_model_path}")

    # If baseline flag is set, evaluate base model on test subjects without finetuning
    if args.baseline:
        print("Evaluating baseline (no fine-tuning) performance on test subjects...")
        # Load the base pretrained model
        base_classifier, metadata = HybridModelClassifier.load_model(base_model_path)
        # Disable CSP to avoid missing CSP transformer
        base_classifier.use_feature_modules = False
        accs = []
        for subject in test_subjects:
            # Load subject-specific data
            X, y, info = load_subject_data(subject_id=subject, run_ids=RUN_IDS, use_motor_channels=USE_MOTOR_CHANNELS)

            # Evaluate baseline accuracy on test set
            results = base_classifier.evaluate(X, y)
            print(f"Baseline accuracy for subject {subject}: {results['accuracy']:.4f}")
            accs.append(results['accuracy'])
        print(f"Mean baseline accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}, {np.min(accs):.4f} - {np.max(accs):.4f}")

    if args.classical:
        print("Evaluating classical Riemannian tangent-space baseline on test subjects...")
        acc = classical_baseline.evaluate_multi_subject_riemannian(train_subjects, test_subjects, RUN_IDS, USE_MOTOR_CHANNELS)
        print(f"Classical baseline accuracy: {acc:.4f}")

    if args.ablation:
        # Run ablation study on validation subjects from the base model
        print("Running ablation study on base model validation subjects...")
        
        # Load the base model
        base_classifier, metadata = HybridModelClassifier.load_model(base_model_path)
        
        assert 'validation_subjects' in metadata, "Validation subjects not found in metadata"
        val_subjects = metadata['validation_subjects']
        print(f"Using {len(val_subjects)} validation subjects from metadata: {val_subjects}")
        
        # Load validation subject data (specify max_subjects if needed)
        X_val, y_val, subject_indices = load_multi_subject_data(
            subject_ids=val_subjects,
            run_ids=RUN_IDS,
            use_motor_channels=USE_MOTOR_CHANNELS
        )
        
        print(f"Loaded validation data: {X_val.shape} samples from {len(val_subjects)} subjects")
        print(f"Class distribution: {np.bincount(y_val)}")
        
        # Provide subject_indices for ablation to build CSP transformer
        base_classifier._ablation_subject_indices = subject_indices
        # Run ablation study on validation data
        csp_transformer = CSPModule(
            n_components=model_spec['n_csp_components'],
            per_subject=True,
            filter_bank=FILTER_BANK_CSP,
            bands=model_spec['csp_bands']
        )
        csp_transformer.fit(X_val, y_val, subject_indices=subject_indices)
        ablation_results = base_classifier.ablation_report(X_val, y_val, csp_transformer=csp_transformer)
        
        # Print results
        print("\nAblation Study Results:")
        for modality, accuracy in ablation_results.items():
            print(f"{modality}: {accuracy:.4f}")

    # Run augmentation strategy sweep with cross-validation if requested
    if args.aug_sweep:
        print("Running augmentation strategy sweep with cross-validation...")
        # Prepare common parameters
        # strategies = [('None', NoAugmenter())]
        # strategies = []
        # for factor in [2, 12]:
        #     tester = StrategyTester(augmentation_factor=factor)
        #     strategies += [
        #         # ('Optimized {factor}', OptimizedAugmenter(augmentation_factor=factor)),
        #         # ('Basic {factor}', BasicAugmenter(augmentation_factor=factor)),
        #         # ('SingleSubject {factor}', SingleSubjectAugmenter(augmentation_factor=factor)),
        #         (f'CustomCombo {factor}', tester.get_combined_strategy_augmenter(
        #             ['add_noise', 'channel_dropout', 'smooth_warping', 'scale_amplitude'],
        #             [1, 1, 1, 1]
        #         ))
        #     ]

        factor = model_spec['augmentation_factor']
        tester = StrategyTester(augmentation_factor=factor)
        strategies = [('None', NoAugmenter())]
        # strategies = []
        strats = tester.available_strategies
        # Add individual strategy augmenters
        for strat in strats:
            strategies.append((strat, tester.get_single_strategy_augmenter(strat)))
        # Add pairs of strategies: both augmentations will be applied to every trial
        for i in range(len(strats)):
            for j in range(i+1, len(strats)):
                name1 = strats[i]
                name2 = strats[j]
                pair_name = f"{name1}+{name2}"
                pair_aug = tester.get_combined_strategy_augmenter([name1, name2], strategy_weights=[1.0, 1.0])
                strategies.append((pair_name, pair_aug))
        # 5-fold over train_subjects list
        kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
        cv_results = {}
        startstring = ""
        for name, _ in strategies:
            print(f"{startstring}{name}", end="")
            startstring = ", "
        print()
        for name, aug in strategies:
            print(f"\n=== Strategy: {name} ===")
            fold_accs = []
            fold_test_accs = []
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_subjects)):
                # Determine train/validation subject lists for this fold
                tr_subjs = [train_subjects[i] for i in train_idx]
                val_subjs = [train_subjects[i] for i in val_idx]
                all_subjs = tr_subjs + val_subjs
                dprint(f" Fold {fold_idx+1}/{kf.n_splits} - Train subjects: {len(tr_subjs)}, Val subjects: {len(val_subjs)}")
                # Perform training on this fold
                classifier, res, _ = train_base_model_multi_subject(
                    run_ids=RUN_IDS,
                    n_subjects=len(all_subjs),
                    subject_ids=all_subjs,
                    val_subject_ids=val_subjs,
                    save_model=False,
                    n_epochs=200,
                    model_name=f"{base_model_name}_{name}_fold{fold_idx+1}",
                    model_spec=model_spec,
                    augmenter=aug,
                    feature_modules=feature_modules
                )
                dprint(f"  Val accuracy: {res['accuracy']:.4f}")
                fold_accs.append(res['accuracy'])
                # Evaluate on held-out test subjects with subject-specific module fitting
                for subject in test_subjects:
                    # Load subject data
                    X_subj, y_subj, _ = load_subject_data(
                        subject_id=subject,
                        run_ids=RUN_IDS,
                        use_motor_channels=USE_MOTOR_CHANNELS
                    )
                    # Stratified split to hold out 25% for evaluation
                    X_fit, X_eval, y_fit, y_eval = train_test_split(
                        X_subj, y_subj,
                        test_size=0.25,
                        stratify=y_subj,
                        random_state=RANDOM_SEED
                    )
                    # Subject indices for a single unseen subject (all zeros)
                    subj_idx_fit = np.zeros(len(y_fit), dtype=int)
                    subj_idx_eval = np.zeros(len(y_eval), dtype=int)
                    # Instantiate and fit fresh feature modules on the fit split
                    test_fms = [
                        CSPModule(
                            n_components=model_spec['n_csp_components'],
                            per_subject=True,
                            filter_bank=FILTER_BANK_CSP,
                            bands=model_spec.get('csp_bands')
                        )
                    ]
                    for fm in test_fms:
                        fm.fit(
                            torch.FloatTensor(X_fit),
                            torch.LongTensor(y_fit),
                            subject_indices=torch.LongTensor(subj_idx_fit)
                        )
                    # Temporarily swap in subject-specific modules
                    orig_fms = classifier.feature_modules
                    classifier.feature_modules = test_fms
                    # Evaluate on the eval split
                    test_res = classifier.evaluate(
                        X_eval, y_eval,
                        subject_indices=subj_idx_eval
                    )
                    dprint(f"  Test accuracy for subject {subject}: {test_res['accuracy']:.4f}")
                    # Restore original modules
                    classifier.feature_modules = orig_fms
                    fold_test_accs.append(test_res['accuracy'])
                # Summary across subjects for this fold
                mean_test = np.mean(fold_test_accs)
                std_test = np.std(fold_test_accs)
                dprint(f"  Mean test accuracy across subjects: {mean_test:.4f} ± {std_test:.4f}")
            mean_acc = np.mean(fold_accs)
            std_acc = np.std(fold_accs)
            mean_test_all = np.mean(fold_test_accs)
            std_test_all = np.std(fold_test_accs)
            print(f"Mean accuracy for {name}: {mean_acc:.4f} ± {std_acc:.4f}, Test: {mean_test_all:.4f} ± {std_test_all:.4f}")
            cv_results[name] = (mean_acc, std_acc, mean_test_all, std_test_all)
        # Summarize CV results sorted by test accuracy
        print("\nAugmentation sweep CV results (sorted by Test Accuracy):")
        # Print header
        print(f"{'Strategy':<30s} {'Val Acc':>8s} {'Val Std':>8s} {'Test Acc':>10s} {'Test Std':>10s}")
        print('-' * 72)
        # Sort results by test accuracy descending
        sorted_results = sorted(cv_results.items(), key=lambda kv: kv[1][2], reverse=True)
        for name, (mean_acc, std_acc, mean_test, std_test) in sorted_results:
            print(f"{name:<30s} {mean_acc:8.4f} {std_acc:8.4f} {mean_test:10.4f} {std_test:10.4f}")

    if args.finetune:
        n_folds = 3
        augmentation_factor = 2
        early_stopping_patience = 30
        augment_percentage = 80
        freeze_layers = True

        # Step 2: Fine-tune for individual test subjects
        fine_tune_results = {}
        test_subjects_progress = tqdm(test_subjects, desc="Test Subjects")
        
        for subject in test_subjects_progress:
            # Fine-tune for this test subject
            test_subjects_progress.set_description(f"Fine-tuning subject {subject}")
            subject_results = finetune_subject_specific(
                base_model_path=base_model_path,
                subject_id=subject,
                run_ids=RUN_IDS,
                n_epochs=200,
                save_models=True,
                val_split=0.25,
                freeze_layers=freeze_layers,
                use_cross_validation=True,
                n_folds=n_folds,
                augmentation_factor=augmentation_factor,
                early_stopping_patience=early_stopping_patience,
                augment_percentage=augment_percentage,
                track_test_acc=True
            )
            fine_tune_results[subject] = subject_results
        
        # After fine-tuning for all subjects, visualize cross-subject results
        if len(fine_tune_results) > 1:  # Only run if we have multiple subjects
            print("\nGenerating cross-subject visualizations and analysis...")
            results_df, strategy_stats, rel_df = visualize_cross_subject_results(fine_tune_results)

    plt.show()