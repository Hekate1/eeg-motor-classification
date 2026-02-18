def evaluate_augmentation_methods(subject_id=SUBJECT_ID, run_ids=RUN_IDS, multi_subject=False, 
                              n_epochs=50, n_repeats=3, save_results=True,
                              evaluation_mode="individual", max_subjects=None):
    """
    Systematically evaluate the impact of different data augmentation methods on
    model performance for single or multi-subject scenarios.
    
    Args:
        subject_id: Subject ID for single-subject evaluation
        run_ids: Run IDs to use
        multi_subject: Whether to evaluate on multiple subjects
        n_epochs: Number of epochs for each training run (reduced for faster evaluation)
        n_repeats: Number of repetitions for each method (to account for randomness)
        save_results: Whether to save the results to a file
        evaluation_mode: Either "individual" (test each method alone) or 
                         "leave_one_out" (test all methods except one)
        max_subjects: Maximum number of subjects to include for multi-subject evaluation
        
    Returns:
        Dictionary containing the evaluation results
    """
    # Load data based on scenario
    if multi_subject:
        print(f"Loading multi-subject data...")
        X, y, subject_indices = load_multi_subject_data(run_ids=run_ids, max_subjects=max_subjects)
        # Apply subject-specific normalization
        X = subject_specific_normalize(X, subject_indices)
        n_subjects = len(np.unique(subject_indices))
        title_prefix = f"Multi-Subject ({n_subjects} subjects)"
        file_prefix = f"multi_{n_subjects}_subjects"
    else:
        print(f"Loading single-subject data for subject {subject_id}...")
        X, y, _ = load_subject_data(subject_id=subject_id, run_ids=run_ids)
        subject_indices = None
        title_prefix = f"Subject {subject_id}"
        file_prefix = f"sub_{subject_id}"
    
    print(f"Data shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Define all augmentation methods available
    all_methods = [
        'add_noise',
        'scale_amplitude',
        'chunk_selection',
        'variable_length_sequence',
        'frequency_band_noise',
        'spectral_perturbation',
        'smooth_warping',
        'trial_mixup'
    ]
    
    # Define augmentation methods to evaluate based on the evaluation mode
    if evaluation_mode == "individual":
        # Test each method individually
        augmentation_methods = [
            {'name': 'None', 'method': None},  # Baseline (no augmentation)
            {'name': 'Noise', 'method': 'add_noise'},
            {'name': 'Scale', 'method': 'scale_amplitude'},
            {'name': 'Chunk', 'method': 'chunk_selection'},
            {'name': 'VarLength', 'method': 'variable_length_sequence'},
            {'name': 'FreqNoise', 'method': 'frequency_band_noise'},
            {'name': 'SpectralPert', 'method': 'spectral_perturbation'},
            {'name': 'SmoothWarp', 'method': 'smooth_warping'},
            {'name': 'Mixup', 'method': 'trial_mixup'},
            {'name': 'All', 'method': 'all'}  # All methods combined
        ]
        mode_suffix = "individual"
    elif evaluation_mode == "leave_one_out":
        # Test the impact of removing each method (leave-one-out)
        augmentation_methods = [
            {'name': 'All', 'method': 'all'},  # All methods (baseline for leave-one-out)
            {'name': 'No Noise', 'method': 'no_noise', 'exclude': 'add_noise'},
            {'name': 'No Scale', 'method': 'no_scale', 'exclude': 'scale_amplitude'},
            {'name': 'No Chunk', 'method': 'no_chunk', 'exclude': 'chunk_selection'},
            {'name': 'No VarLength', 'method': 'no_varlength', 'exclude': 'variable_length_sequence'},
            {'name': 'No FreqNoise', 'method': 'no_freqnoise', 'exclude': 'frequency_band_noise'},
            {'name': 'No SpectralPert', 'method': 'no_spectralpert', 'exclude': 'spectral_perturbation'},
            {'name': 'No SmoothWarp', 'method': 'no_smoothwarp', 'exclude': 'smooth_warping'},
            {'name': 'No Mixup', 'method': 'no_mixup', 'exclude': 'trial_mixup'},
            {'name': 'None', 'method': None}  # No augmentation (for reference)
        ]
        mode_suffix = "leave_one_out"
    else:
        raise ValueError(f"Unknown evaluation mode: {evaluation_mode}")
    
    # Create a custom augmenter class for method isolation or exclusion
    class CustomAugmenter(EEGDataAugmenter):
        def __init__(self, method_name=None, exclude_method=None, noise_level=0.03, 
                     scale_range=(0.9, 1.1), chunk_max_offset=30):
            super().__init__(noise_level, scale_range, chunk_max_offset, support_variable_length=False)
            self.method_name = method_name
            self.exclude_method = exclude_method
            
            # Available augmentation methods
            self.available_methods = {
                'add_noise': self.add_noise,
                'scale_amplitude': self.scale_amplitude,
                'chunk_selection': self.chunk_selection,
                'variable_length_sequence': self.variable_length_sequence,
                'frequency_band_noise': self.frequency_band_noise,
                'spectral_perturbation': self.spectral_perturbation,
                'smooth_warping': self.smooth_warping
                # trial_mixup is handled separately
            }
            
        def augment(self, x, y=None):
            """Apply specific augmentation method(s) based on the evaluation mode"""
            # Handle the None case (no augmentation)
            if self.method_name is None:
                return x.copy()
                
            # Individual method mode
            if self.exclude_method is None:
                if self.method_name == 'add_noise':
                    return self.add_noise(x.copy())
                elif self.method_name == 'scale_amplitude':
                    return self.scale_amplitude(x.copy())
                elif self.method_name == 'chunk_selection':
                    return self.chunk_selection(x.copy())
                elif self.method_name == 'variable_length_sequence':
                    return self.variable_length_sequence(x.copy())
                elif self.method_name == 'frequency_band_noise':
                    return self.frequency_band_noise(x.copy())
                elif self.method_name == 'spectral_perturbation':
                    return self.spectral_perturbation(x.copy())
                elif self.method_name == 'smooth_warping':
                    return self.smooth_warping(x.copy())
                elif self.method_name == 'trial_mixup':
                    return self.trial_mixup(x.copy(), y)
                elif self.method_name == 'all':
                    # Apply 2-3 random augmentations from the full list
                    x_aug = x.copy()
                    
                    # Get all available augmentation methods
                    augmentations = list(self.available_methods.values())
                    
                    # Apply 2-3 random augmentations
                    num_augmentations = np.random.randint(2, 4)
                    selected_augmentations = np.random.choice(augmentations, 
                                                            size=min(num_augmentations, len(augmentations)), 
                                                            replace=False)
                    
                    # Apply selected augmentations
                    for augmentation in selected_augmentations:
                        x_aug = augmentation(x_aug)
                    
                    # Separately decide whether to apply mixup (with 30% probability)
                    if self.trial_cache is not None and np.random.rand() < 0.3:
                        x_aug = self.trial_mixup(x_aug, y, same_class_only=True)
                        
                    return x_aug
                else:
                    return x.copy()
            # Leave-one-out mode
            else:
                x_aug = x.copy()
                
                # Create a copy of available methods and remove the excluded one
                available_methods = dict(self.available_methods)
                if self.exclude_method in available_methods:
                    del available_methods[self.exclude_method]
                
                # Get the list of augmentation functions that are available
                augmentations = list(available_methods.values())
                
                # Apply 2-3 random augmentations from the filtered list
                num_augmentations = np.random.randint(2, 4)
                selected_augmentations = np.random.choice(augmentations, 
                                                        size=min(num_augmentations, len(augmentations)), 
                                                        replace=False)
                
                # Apply selected augmentations
                for augmentation in selected_augmentations:
                    x_aug = augmentation(x_aug)
                
                # Apply mixup separately (unless it's the excluded method)
                if self.exclude_method != 'trial_mixup' and self.trial_cache is not None and np.random.rand() < 0.3:
                    x_aug = self.trial_mixup(x_aug, y)
                    
                return x_aug
                
        def augment_batch(self, X, y, augmentation_factor=2):
            """Generate augmented data for a batch of EEG trials"""
            n_trials = X.shape[0]
            
            # Cache a copy of the original trials for mixup augmentation
            self.trial_cache = [X[i].copy() for i in range(n_trials)]
            if y is not None:
                self.trial_labels = y.copy()
            
            # Fixed-length mode (we're not supporting variable-length for evaluation)
            augmented_X = [X]
            augmented_y = [y]
            
            # Add augmented data
            for _ in range(augmentation_factor):
                X_batch_aug = np.array([self.augment(X[i], y[i] if y is not None else None) for i in range(n_trials)])
                augmented_X.append(X_batch_aug)
                augmented_y.append(y)
            
            # Concatenate original and augmented data
            X_augmented = np.concatenate(augmented_X, axis=0)
            y_augmented = np.concatenate(augmented_y, axis=0)
            
            return X_augmented, y_augmented
    
    # Create a fixed train/val split to ensure fair comparison
    n_classes = len(np.unique(y))
    indices_by_class = [np.where(y == i)[0] for i in range(n_classes)]
    
    train_indices = []
    val_indices = []
    
    # Use a fixed random seed for reproducibility
    np.random.seed(42)
    
    for class_indices in indices_by_class:
        np.random.shuffle(class_indices)
        val_size = int(len(class_indices) * 0.2)
        
        val_indices.extend(class_indices[:val_size])
        train_indices.extend(class_indices[val_size:])
    
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    
    # Initialize results storage
    results = {
        'method_names': [m['name'] for m in augmentation_methods],
        'val_accuracy': np.zeros((len(augmentation_methods), n_repeats)),
        'val_balanced_accuracy': np.zeros((len(augmentation_methods), n_repeats)),
        'training_time': np.zeros((len(augmentation_methods), n_repeats)),
        'best_epoch': np.zeros((len(augmentation_methods), n_repeats), dtype=int)
    }
    
    # Function to create a classifier with a specific augmenter
    def create_classifier(method_info):
        # Consistent batch size across all methods
        batch_size = 16
    
        if multi_subject:
            kwargs = {
                'n_classes': n_classes,
                'embedding_dim': 64,
                'n_heads': 4,
                'n_layers': 2,
                'dropout': 0.3,
                'lr': 0.0005,
                'batch_size': batch_size,
                'n_epochs': n_epochs,
                'weight_decay': 0.01,
                'augmentation_factor': 3,  # Increase augmentation factor for more data
                'use_csp': True,
                'use_freq': True,
                'n_csp_components': 6,
                'variable_length_augmentation': False  # Disable variable length for evaluation
            }
        else:
            kwargs = {
                'n_classes': n_classes,
                'embedding_dim': 64,
                'n_heads': 4,
                'n_layers': 2,
                'dropout': 0.3,
                'lr': 0.0005,
                'batch_size': batch_size,
                'n_epochs': n_epochs,
                'weight_decay': 0.01,
                'augmentation_factor': 3,  # Increase augmentation factor for more data
                'use_csp': True,
                'use_freq': True,
                'n_csp_components': 4,
                'variable_length_augmentation': False  # Disable variable length for evaluation
            }
        
        classifier = HybridModelClassifier(**kwargs)
        
        # Replace the augmenter with our custom version
        if evaluation_mode == "individual":
            classifier.augmenter = CustomAugmenter(method_name=method_info['method'])
        else:  # leave_one_out
            exclude = method_info.get('exclude', None)
            classifier.augmenter = CustomAugmenter(method_name=method_info['method'], 
                                                  exclude_method=exclude)
        
        # Cache the training data for mixup augmentation
        # This will happen in prepare_data
        
        return classifier
    
    # Evaluate each augmentation method
    for i, aug_method in enumerate(augmentation_methods):
        method_name = aug_method['name']
        print(f"\n{'='*50}")
        print(f"Evaluating: {method_name}")
        print(f"{'='*50}")
        
        for repeat in range(n_repeats):
            print(f"\nRepetition {repeat+1}/{n_repeats}")
            
            # Create classifier with the specific augmentation method
            classifier = create_classifier(aug_method)
            
            # Track training time
            start_time = time.time()
            
            # Train and evaluate
            try:
                train_results = classifier.train_and_evaluate(
                    X, y, val_split=0.2, early_stopping_patience=20  # Increased patience
                )
                
                # Store results (these use the best model from training)
                results['val_accuracy'][i, repeat] = train_results['accuracy']
                results['val_balanced_accuracy'][i, repeat] = train_results['balanced_accuracy']
                
                # Record the best epoch
                best_epoch = np.argmax(train_results['history']['val_balanced_acc'])
                results['best_epoch'][i, repeat] = best_epoch + 1  # +1 since epochs are 1-indexed in output
            except Exception as e:
                print(f"Error during training: {str(e)}")
                results['val_accuracy'][i, repeat] = np.nan
                results['val_balanced_accuracy'][i, repeat] = np.nan
                results['best_epoch'][i, repeat] = 0
            
            end_time = time.time()
            results['training_time'][i, repeat] = end_time - start_time
            
            print(f"Validation accuracy: {results['val_accuracy'][i, repeat]:.4f}")
            print(f"Validation balanced accuracy: {results['val_balanced_accuracy'][i, repeat]:.4f}")
            print(f"Best epoch: {results['best_epoch'][i, repeat]}")
            print(f"Training time: {results['training_time'][i, repeat]:.1f} seconds")
    
    # Calculate average and standard deviation
    results['mean_accuracy'] = np.nanmean(results['val_accuracy'], axis=1)
    results['std_accuracy'] = np.nanstd(results['val_accuracy'], axis=1)
    results['mean_balanced_accuracy'] = np.nanmean(results['val_balanced_accuracy'], axis=1)
    results['std_balanced_accuracy'] = np.nanstd(results['val_balanced_accuracy'], axis=1)
    results['mean_training_time'] = np.nanmean(results['training_time'], axis=1)
    results['mean_best_epoch'] = np.nanmean(results['best_epoch'], axis=1)
    
    # Visualize results
    plt.figure(figsize=(12, 12))
    
    # Plot accuracy
    plt.subplot(3, 1, 1)
    x = np.arange(len(results['method_names']))
    width = 0.35
    
    plt.bar(x - width/2, results['mean_accuracy'], width, 
            yerr=results['std_accuracy'], label='Accuracy', 
            color='royalblue', alpha=0.7, capsize=5)
    plt.bar(x + width/2, results['mean_balanced_accuracy'], width, 
            yerr=results['std_balanced_accuracy'], label='Balanced Accuracy', 
            color='darkorange', alpha=0.7, capsize=5)
    
    plt.xlabel('Augmentation Method')
    plt.ylabel('Validation Accuracy')
    if evaluation_mode == "leave_one_out":
        plt.title(f'{title_prefix} - Impact of Removing Each Augmentation Method')
    else:
        plt.title(f'{title_prefix} - Impact of Individual Augmentation Methods')
    plt.xticks(x, results['method_names'], rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot training time
    plt.subplot(3, 1, 2)
    plt.bar(x, results['mean_training_time'], color='forestgreen', alpha=0.7)
    plt.xlabel('Augmentation Method')
    plt.ylabel('Training Time (s)')
    plt.title(f'{title_prefix} - Training Time by Method')
    plt.xticks(x, results['method_names'], rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot best epoch
    plt.subplot(3, 1, 3)
    plt.bar(x, results['mean_best_epoch'], color='purple', alpha=0.7)
    plt.xlabel('Augmentation Method')
    plt.ylabel('Best Epoch')
    plt.title(f'{title_prefix} - Average Best Epoch by Method')
    plt.xticks(x, results['method_names'], rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save results
    if save_results:
        # Create results directory if it doesn't exist
        results_dir = os.path.join(os.getcwd(), 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save plot
        plt.savefig(os.path.join(results_dir, f'augmentation_impact_{file_prefix}_{mode_suffix}.png'))
        
        # Save numerical results
        np.savez(os.path.join(results_dir, f'augmentation_impact_{file_prefix}_{mode_suffix}.npz'), **results)
    
    
    # Print summary
    print("\nSummary of Augmentation Methods Impact:")
    print(f"{'Method':<15} {'Accuracy':<15} {'Balanced Acc':<15} {'Time (s)':<10} {'Best Epoch':<10}")
    print('-' * 65)
    
    for i, method in enumerate(results['method_names']):
        print(f"{method:<15} {results['mean_accuracy'][i]:.4f} ± {results['std_accuracy'][i]:.4f}   "
              f"{results['mean_balanced_accuracy'][i]:.4f} ± {results['std_balanced_accuracy'][i]:.4f}   "
              f"{results['mean_training_time'][i]:.1f}      {results['mean_best_epoch'][i]:.1f}")
    
    # Find the best method
    if evaluation_mode == "individual":
        best_idx = np.argmax(results['mean_balanced_accuracy'])
        best_method = results['method_names'][best_idx]
        print(f"\nBest augmentation method: {best_method} "
              f"(Balanced Accuracy: {results['mean_balanced_accuracy'][best_idx]:.4f})")
    else:  # leave_one_out
        # For leave-one-out, the worst method to exclude (causing biggest drop in performance) is most important
        all_idx = results['method_names'].index('All')
        all_acc = results['mean_balanced_accuracy'][all_idx]
        
        # Calculate performance drops when excluding each method
        drops = []
        for i, name in enumerate(results['method_names']):
            if name.startswith('No '):
                drop = all_acc - results['mean_balanced_accuracy'][i]
                drops.append((name[3:], drop))  # Remove "No " prefix
        
        # Sort by largest performance drop
        drops.sort(key=lambda x: x[1], reverse=True)
        
        # Print the most important methods (those with largest drop when excluded)
        print("\nMost important augmentation methods (ranked by performance drop when excluded):")
        for method, drop in drops:
            print(f"{method}: {drop:.4f} drop in balanced accuracy when excluded")
    
    return results


def visualize_augmentation_effects(subject_id=SUBJECT_ID, run_ids=RUN_IDS, n_examples=5):
    """
    Visualize the effect of each augmentation method on EEG data
    
    Args:
        subject_id: Subject ID to use
        run_ids: Run IDs to use
        n_examples: Number of example trials to visualize
    """
    # Load data
    X, y, _ = load_subject_data(subject_id=subject_id, run_ids=run_ids)
    
    # Apply normalization to better see the effects
    X = subject_specific_normalize(X)
    
    # Create augmenter
    augmenter = EEGDataAugmenter()
    
    # Get augmentation methods
    augmentation_methods = [
        {'name': 'Original', 'method': None},
        {'name': 'Noise', 'method': augmenter.add_noise},
        {'name': 'Scale', 'method': augmenter.scale_amplitude},
        {'name': 'Chunk', 'method': augmenter.chunk_selection},
        {'name': 'VarLength', 'method': augmenter.variable_length_sequence},
        {'name': 'FreqNoise', 'method': augmenter.frequency_band_noise},
        {'name': 'SpectralPert', 'method': augmenter.spectral_perturbation},
        {'name': 'SmoothWarp', 'method': augmenter.smooth_warping}
        # Mixup requires multiple trials, handled separately below
    ]
    
    # Cache some trials for mixup visualization
    augmenter.trial_cache = [X[i].copy() for i in range(min(20, len(X)))]
    augmenter.trial_labels = y[:min(20, len(y))].copy()
    
    # Select random examples
    n_trials = X.shape[0]
    example_indices = np.random.choice(n_trials, size=n_examples, replace=False)
    
    for idx in example_indices:
        # Get original sample
        x_orig = X[idx]
        label = y[idx]
        
        # Apply each augmentation method
        augmented_samples = [x_orig.copy()]
        for method in augmentation_methods[1:]:
            if method['method'] is not None:
                augmented_samples.append(method['method'](x_orig.copy()))
        
        # Add mixup as a special case
        if augmenter.trial_cache is not None:
            augmented_samples.append(augmenter.trial_mixup(x_orig.copy(), label))
            method_names = [m['name'] for m in augmentation_methods] + ['Mixup']
        else:
            method_names = [m['name'] for m in augmentation_methods]
        
        # Plot
        plt.figure(figsize=(15, 12))
        
        # Plot central electrode (assuming C3 or similar is in the middle)
        center_ch = x_orig.shape[0] // 2
        
        for i, (x_aug, method_name) in enumerate(zip(augmented_samples, method_names)):
            plt.subplot(len(augmented_samples), 1, i+1)
            plt.plot(x_aug[center_ch])
            plt.title(f"{method_name}")
            plt.ylabel('Amplitude')
            
            # For first subplot add more details
            if i == 0:
                plt.title(f"Original Signal (Class {label}, Trial {idx}, Channel {center_ch})")
        
        plt.xlabel('Time points')
        plt.tight_layout()
        
        # Plot frequency domain representation
        plt.figure(figsize=(15, 12))
        
        for i, (x_aug, method_name) in enumerate(zip(augmented_samples, method_names)):
            plt.subplot(len(augmented_samples), 1, i+1)
            
            # Apply FFT
            x_fft = np.abs(np.fft.rfft(x_aug[center_ch]))
            freqs = np.fft.rfftfreq(x_aug.shape[1], d=1/160)  # Assuming 160Hz sampling rate
            
            plt.semilogy(freqs, x_fft)
            plt.title(f"{method_name} - Frequency Domain")
            plt.ylabel('Log Power')
            
            # Highlight mu and beta bands
            plt.axvspan(8, 12, color='yellow', alpha=0.3, label='μ band')
            plt.axvspan(13, 30, color='green', alpha=0.3, label='β band')
            
            if i == 0:
                plt.legend()
        
        plt.xlabel('Frequency (Hz)')
        plt.tight_layout()


def evaluate_optimized_augmentation(subject_id=SUBJECT_ID, run_ids=RUN_IDS, multi_subject=True, 
                               n_epochs=150, n_repeats=5, save_results=True, max_subjects=None):
    """
    Evaluate a custom optimized augmentation strategy that uses only the beneficial methods
    based on leave-one-out analysis results.
    
    Args:
        subject_id: Subject ID for single-subject evaluation
        run_ids: Run IDs to use
        multi_subject: Whether to evaluate on multiple subjects
        n_epochs: Number of epochs for each training run
        n_repeats: Number of repetitions for each method
        save_results: Whether to save the results to a file
        max_subjects: Maximum number of subjects to include (None = all)
        
    Returns:
        Dictionary containing the evaluation results
    """
    # Load data based on scenario
    if multi_subject:
        print(f"Loading multi-subject data...")
        X, y, subject_indices = load_multi_subject_data(run_ids=run_ids, max_subjects=max_subjects)
        # Apply subject-specific normalization
        X = subject_specific_normalize(X, subject_indices)
        n_subjects = len(np.unique(subject_indices))
        title_prefix = f"Multi-Subject ({n_subjects} subjects)"
        file_prefix = f"multi_{n_subjects}_subjects"
    else:
        print(f"Loading single-subject data for subject {subject_id}...")
        X, y, _ = load_subject_data(subject_id=subject_id, run_ids=run_ids)
        subject_indices = None
        title_prefix = f"Subject {subject_id}"
        file_prefix = f"sub_{subject_id}"
    
    print(f"Data shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Define augmentation methods to evaluate
    augmentation_methods = [
        {'name': 'None', 'method': None},  # Baseline (no augmentation)
        {'name': 'Optimized', 'method': 'optimized'},  # Our optimized strategy
        {'name': 'All', 'method': 'all'}  # All methods (for comparison)
    ]
    
    # Create a custom augmenter class for optimized augmentation
    class CustomOptimizedAugmenter(OptimizedAugmenter):
        def __init__(self, method_name=None, noise_level=0.03, 
                     scale_range=(0.9, 1.1), chunk_max_offset=30):
            super().__init__(noise_level, scale_range, chunk_max_offset, support_variable_length=False)
            self.method_name = method_name
            
            # All available methods (for comparison)
            self.all_methods = {
                'add_noise': self.add_noise,
                'scale_amplitude': self.scale_amplitude,
                'chunk_selection': self.chunk_selection,
                'variable_length_sequence': self.variable_length_sequence,
                'frequency_band_noise': self.frequency_band_noise,
                'spectral_perturbation': self.spectral_perturbation,
                'smooth_warping': self.smooth_warping
            }
            
        def augment(self, x, y=None):
            """Apply specific augmentation method(s) based on the strategy"""
            # Handle the None case (no augmentation)
            if self.method_name is None:
                return x.copy()
                
            # Optimized strategy
            if self.method_name == 'optimized':
                # Use our parent class implementation for 'optimized' method
                return super().augment(x, y)
                
            # All methods (for comparison)
            elif self.method_name == 'all':
                x_aug = x.copy()
                
                # Get all available augmentation methods
                augmentations = list(self.all_methods.values())
                
                # Apply 2-3 random augmentations
                num_augmentations = np.random.randint(2, 4)
                selected_augmentations = np.random.choice(augmentations, 
                                                        size=min(num_augmentations, len(augmentations)), 
                                                        replace=False)
                
                # Apply selected augmentations
                for augmentation in selected_augmentations:
                    x_aug = augmentation(x_aug)
                
                # Apply mixup with standard probability (30%)
                if self.trial_cache is not None and np.random.rand() < 0.3:
                    x_aug = self.trial_mixup(x_aug, y, same_class_only=True)
                    
                return x_aug
            else:
                return x.copy()
                
        def augment_batch(self, X, y, augmentation_factor=3):
            """Generate augmented data for a batch of EEG trials"""
            n_trials = X.shape[0]
            
            # Cache a copy of the original trials for mixup augmentation
            self.trial_cache = [X[i].copy() for i in range(n_trials)]
            if y is not None:
                self.trial_labels = y.copy()
            
            # Fixed-length mode
            augmented_X = [X]
            augmented_y = [y]
            
            # Add augmented data
            for _ in range(augmentation_factor):
                X_batch_aug = np.array([self.augment(X[i], y[i] if y is not None else None) for i in range(n_trials)])
                augmented_X.append(X_batch_aug)
                augmented_y.append(y)
            
            # Concatenate original and augmented data
            X_augmented = np.concatenate(augmented_X, axis=0)
            y_augmented = np.concatenate(augmented_y, axis=0)
            
            return X_augmented, y_augmented
    
    # Create a fixed train/val split to ensure fair comparison
    n_classes = len(np.unique(y))
    indices_by_class = [np.where(y == i)[0] for i in range(n_classes)]
    
    train_indices = []
    val_indices = []
    
    # Use a fixed random seed for reproducibility
    np.random.seed(42)
    
    for class_indices in indices_by_class:
        np.random.shuffle(class_indices)
        val_size = int(len(class_indices) * 0.2)
        
        val_indices.extend(class_indices[:val_size])
        train_indices.extend(class_indices[val_size:])
    
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    
    # Initialize results storage
    results = {
        'method_names': [m['name'] for m in augmentation_methods],
        'val_accuracy': np.zeros((len(augmentation_methods), n_repeats)),
        'val_balanced_accuracy': np.zeros((len(augmentation_methods), n_repeats)),
        'training_time': np.zeros((len(augmentation_methods), n_repeats)),
        'best_epoch': np.zeros((len(augmentation_methods), n_repeats), dtype=int)
    }
    
    # Function to create a classifier with the optimized augmenter
    def create_classifier(method_info):
        # Consistent batch size across all methods
        batch_size = 24  # Increased batch size for more stability
    
        if multi_subject:
            kwargs = {
                'n_classes': n_classes,
                'embedding_dim': 64,
                'n_heads': 4,
                'n_layers': 2,
                'dropout': 0.3,
                'lr': 0.0004,  # Slightly lower learning rate
                'batch_size': batch_size,
                'n_epochs': n_epochs,
                'weight_decay': 0.01,
                'augmentation_factor': 3,  # Enhanced augmentation
                'use_csp': True,
                'use_freq': True,
                'n_csp_components': 6,
                'variable_length_augmentation': False
            }
        else:
            kwargs = {
                'n_classes': n_classes,
                'embedding_dim': 64,
                'n_heads': 4,
                'n_layers': 2,
                'dropout': 0.3,
                'lr': 0.0004,  # Slightly lower learning rate
                'batch_size': batch_size,
                'n_epochs': n_epochs,
                'weight_decay': 0.01,
                'augmentation_factor': 3,  # Enhanced augmentation
                'use_csp': True,
                'use_freq': True,
                'n_csp_components': 4,
                'variable_length_augmentation': False
            }
        
        classifier = HybridModelClassifier(**kwargs)
        
        # Replace the augmenter with our optimized version
        classifier.augmenter = CustomOptimizedAugmenter(method_name=method_info['method'])
        
        return classifier
    
    # Evaluate each augmentation strategy
    for i, aug_method in enumerate(augmentation_methods):
        method_name = aug_method['name']
        print(f"\n{'='*50}")
        print(f"Evaluating: {method_name}")
        print(f"{'='*50}")
        
        for repeat in range(n_repeats):
            print(f"\nRepetition {repeat+1}/{n_repeats}")
            
            # Create classifier with the specific augmentation method
            classifier = create_classifier(aug_method)
            
            # Track training time
            start_time = time.time()
            
            # Train and evaluate
            try:
                train_results = classifier.train_and_evaluate(
                    X, y, val_split=0.2, early_stopping_patience=35  # Increased patience
                )
                
                # Store results (these use the best model from training)
                results['val_accuracy'][i, repeat] = train_results['accuracy']
                results['val_balanced_accuracy'][i, repeat] = train_results['balanced_accuracy']
                
                # Record the best epoch
                best_epoch = np.argmax(train_results['history']['val_balanced_acc'])
                results['best_epoch'][i, repeat] = best_epoch + 1  # +1 since epochs are 1-indexed in output
            except Exception as e:
                print(f"Error during training: {str(e)}")
                results['val_accuracy'][i, repeat] = np.nan
                results['val_balanced_accuracy'][i, repeat] = np.nan
                results['best_epoch'][i, repeat] = 0
            
            end_time = time.time()
            results['training_time'][i, repeat] = end_time - start_time
            
            print(f"Validation accuracy: {results['val_accuracy'][i, repeat]:.4f}")
            print(f"Validation balanced accuracy: {results['val_balanced_accuracy'][i, repeat]:.4f}")
            print(f"Best epoch: {results['best_epoch'][i, repeat]}")
            print(f"Training time: {results['training_time'][i, repeat]:.1f} seconds")
    
    # Calculate average and standard deviation
    results['mean_accuracy'] = np.nanmean(results['val_accuracy'], axis=1)
    results['std_accuracy'] = np.nanstd(results['val_accuracy'], axis=1)
    results['mean_balanced_accuracy'] = np.nanmean(results['val_balanced_accuracy'], axis=1)
    results['std_balanced_accuracy'] = np.nanstd(results['val_balanced_accuracy'], axis=1)
    results['mean_training_time'] = np.nanmean(results['training_time'], axis=1)
    results['mean_best_epoch'] = np.nanmean(results['best_epoch'], axis=1)
    
    # Visualize results
    plt.figure(figsize=(12, 12))
    
    # Plot accuracy
    plt.subplot(3, 1, 1)
    x = np.arange(len(results['method_names']))
    width = 0.35
    
    plt.bar(x - width/2, results['mean_accuracy'], width, 
            yerr=results['std_accuracy'], label='Accuracy', 
            color='royalblue', alpha=0.7, capsize=5)
    plt.bar(x + width/2, results['mean_balanced_accuracy'], width, 
            yerr=results['std_balanced_accuracy'], label='Balanced Accuracy', 
            color='darkorange', alpha=0.7, capsize=5)
    
    plt.xlabel('Augmentation Strategy')
    plt.ylabel('Validation Accuracy')
    plt.title(f'{title_prefix} - Optimized vs. Standard Augmentation')
    plt.xticks(x, results['method_names'], rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot training time
    plt.subplot(3, 1, 2)
    plt.bar(x, results['mean_training_time'], color='forestgreen', alpha=0.7)
    plt.xlabel('Augmentation Strategy')
    plt.ylabel('Training Time (s)')
    plt.title(f'{title_prefix} - Training Time by Strategy')
    plt.xticks(x, results['method_names'], rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot best epoch
    plt.subplot(3, 1, 3)
    plt.bar(x, results['mean_best_epoch'], color='purple', alpha=0.7)
    plt.xlabel('Augmentation Strategy')
    plt.ylabel('Best Epoch')
    plt.title(f'{title_prefix} - Average Best Epoch by Strategy')
    plt.xticks(x, results['method_names'], rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save results
    if save_results:
        # Create results directory if it doesn't exist
        results_dir = os.path.join(os.getcwd(), 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save plot
        plt.savefig(os.path.join(results_dir, f'optimized_augmentation_{file_prefix}.png'))
        
        # Save numerical results
        np.savez(os.path.join(results_dir, f'optimized_augmentation_{file_prefix}.npz'), **results)
    
    # Print summary
    print("\nSummary of Augmentation Strategies:")
    print(f"{'Strategy':<15} {'Accuracy':<15} {'Balanced Acc':<15} {'Time (s)':<10} {'Best Epoch':<10}")
    print('-' * 65)
    
    for i, method in enumerate(results['method_names']):
        print(f"{method:<15} {results['mean_accuracy'][i]:.4f} ± {results['std_accuracy'][i]:.4f}   "
              f"{results['mean_balanced_accuracy'][i]:.4f} ± {results['std_balanced_accuracy'][i]:.4f}   "
              f"{results['mean_training_time'][i]:.1f}      {results['mean_best_epoch'][i]:.1f}")
    
    # Identify best strategy
    best_idx = np.argmax(results['mean_balanced_accuracy'])
    best_method = results['method_names'][best_idx]
    print(f"\nBest augmentation strategy: {best_method} "
          f"(Balanced Accuracy: {results['mean_balanced_accuracy'][best_idx]:.4f})")
    
    return results

