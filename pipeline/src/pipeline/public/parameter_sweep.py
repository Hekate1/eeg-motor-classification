import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import torch
from tqdm import tqdm

# Import functions from run.py
# These need to be imported when this module is used
# Run this with "from run import *" before using functions in this module
RUN_IDS = None  # Will be set from run.py
finetune_subject_specific = None  # Will be set from run.py

def run_parameter_sweep(base_model_path, subject_ids, param_grid, run_ids=None):
    """
    Run a parameter sweep to test different combinations of hyperparameters.
    
    Args:
        base_model_path: Path to the pretrained model
        subject_ids: List of subject IDs to test on
        param_grid: Dictionary with parameter names as keys and lists of values to test as values
        run_ids: Run IDs to use
        
    Returns:
        Dictionary of results for all parameter combinations
    """
    global RUN_IDS
    if run_ids is not None:
        RUN_IDS = run_ids
        
    print(f"Running parameter sweep with {len(subject_ids)} subjects")
    
    # Create all combinations of parameters
    from itertools import product
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    
    # Create directory for sweep results
    results_dir = os.path.join(os.getcwd(), 'plots/sweep_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Dictionary to store results for all combinations
    all_results = {}
    
    # Progress tracking
    total_combinations = len(param_combinations)
    print(f"Testing {total_combinations} parameter combinations")
    
    for i, combination in enumerate(param_combinations):
        # Create parameter dictionary for this combination
        params = dict(zip(param_names, combination))
        combo_name = "|".join([f"{k}={v}" for k, v in params.items()])
        
        print(f"\nCombination {i+1}/{total_combinations}: {combo_name}")
        print("-" * 80)
        
        # Dictionary to store results for this combination
        combo_results = {}
        
        # Test this combination on all subjects
        for subject_id in subject_ids:
            print(f"Testing subject {subject_id} with {combo_name}")
            
            # Run fine-tuning with the specified parameters
            results = finetune_subject_specific(
                base_model_path=base_model_path,
                subject_id=subject_id,
                run_ids=RUN_IDS,
                n_epochs=params.get('n_epochs', 200),
                save_models=False,  # Don't save models during parameter sweep
                val_split=params.get('val_split', 0.2),
                freeze_layers=params.get('freeze_layers', True),
                use_cross_validation=params.get('use_cross_validation', True),
                n_folds=params.get('n_folds', 5),
                augmentation_factor=params.get('augmentation_factor', 10),
                early_stopping_patience=params.get('early_stopping_patience', 15),
                augment_percentage=params.get('augment_percentage', 50),
                n_csp_components=params.get('n_csp_components', 6)
            )
            
            # Store results for this subject
            combo_results[subject_id] = results
        
        # Store results for this combination
        all_results[combo_name] = combo_results
        
        # Calculate average performance for this combination
        val_accs = []
        test_accs = []
        
        for subject_id, subject_results in combo_results.items():
            # Find the best strategy for each subject
            best_idx = np.argmax(subject_results['accuracy'])
            val_accs.append(subject_results['accuracy'][best_idx])
            test_accs.append(subject_results['test_accuracy'][best_idx])
        
        avg_val_acc = np.mean(val_accs)
        avg_test_acc = np.mean(test_accs)
        
        print(f"Combination {combo_name}:")
        print(f"Average validation accuracy: {avg_val_acc:.4f}")
        print(f"Average test accuracy: {avg_test_acc:.4f}")
        
        # Save intermediate results
        sweep_results_path = os.path.join(results_dir, f'sweep_results_{i+1}of{total_combinations}.pkl')
        with open(sweep_results_path, 'wb') as f:
            import pickle
            pickle.dump({
                'params': params,
                'combo_name': combo_name,
                'results': combo_results,
                'avg_val_acc': avg_val_acc,
                'avg_test_acc': avg_test_acc
            }, f)
    
    # Analyze and visualize results
    analyze_parameter_sweep(all_results, param_grid)
    
    return all_results


def analyze_parameter_sweep(sweep_results, param_grid):
    """
    Analyze and visualize the results of a parameter sweep.
    
    Args:
        sweep_results: Dictionary of results from run_parameter_sweep
        param_grid: Original parameter grid used for the sweep
    """
    # Create directory for visualizations
    viz_dir = os.path.join(os.getcwd(), 'plots/sweep_visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Extract parameters and performance metrics
    data = []
    
    for combo_name, combo_results in sweep_results.items():
        # Parse parameters from combo_name
        params = {}
        for param_part in combo_name.split('|'):                
            name, value = param_part.split('=')
            # Try to convert to numeric if possible
            try:
                value = float(value)
                if value.is_integer():
                    value = int(value)
            except:
                pass
            params[name] = value
        
        # Calculate average performance across subjects
        val_accs = []
        test_accs = []
        
        for subject_id, subject_results in combo_results.items():
            # Find the best strategy for each subject
            best_idx = np.argmax(subject_results['accuracy'])
            val_accs.append(subject_results['accuracy'][best_idx])
            test_accs.append(subject_results['test_accuracy'][best_idx])
        
        avg_val_acc = np.mean(val_accs)
        avg_test_acc = np.mean(test_accs)
        
        # Add to data
        entry = params.copy()
        entry['avg_val_acc'] = avg_val_acc
        entry['avg_test_acc'] = avg_test_acc
        data.append(entry)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(data)
    
    # Save raw results
    results_df.to_csv(os.path.join(viz_dir, 'parameter_sweep_results.csv'), index=False)
    
    # Create visualizations based on the parameters
    # For parameters with multiple values, create line or heatmap plots
    
    # If we have exactly two parameters with multiple values, create a heatmap
    multi_params = [param for param, values in param_grid.items() if len(values) > 1]
    
    if len(multi_params) == 2:
        # Create heatmap
        param1, param2 = multi_params
        
        # Validation accuracy heatmap
        plt.figure(figsize=(10, 8))
        heatmap_data = results_df.pivot(index=param1, columns=param2, values='avg_val_acc')
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis')
        plt.title(f'Validation Accuracy by {param1} and {param2}')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f'val_acc_heatmap_{param1}_{param2}.png'))
        
        # Test accuracy heatmap
        plt.figure(figsize=(10, 8))
        heatmap_data = results_df.pivot(index=param1, columns=param2, values='avg_test_acc')
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis')
        plt.title(f'Test Accuracy by {param1} and {param2}')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f'test_acc_heatmap_{param1}_{param2}.png'))
    
    # For each parameter, create a line plot
    for param in multi_params:
        plt.figure(figsize=(10, 6))
        
        # Group by this parameter and calculate mean and std
        grouped = results_df.groupby(param)[['avg_val_acc', 'avg_test_acc']].agg(['mean', 'std']).reset_index()
        
        # Make column names more accessible
        grouped.columns = ['_'.join(col) if col[1] else col[0] for col in grouped.columns]
        
        # Plot
        plt.errorbar(grouped[param], grouped['avg_val_acc_mean'], 
                    yerr=grouped['avg_val_acc_std'], 
                    marker='o', label='Validation')
        plt.errorbar(grouped[param], grouped['avg_test_acc_mean'], 
                    yerr=grouped['avg_test_acc_std'], 
                    marker='s', label='Test')
        
        plt.title(f'Performance by {param}')
        plt.xlabel(param)
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f'performance_by_{param}.png'))
    
    # Create summary table
    print("\nParameter Sweep Results Summary:")
    print("-" * 80)
    
    # Sort by test accuracy
    results_df = results_df.sort_values('avg_test_acc', ascending=False)
    
    # Format parameters for display
    param_columns = list(param_grid.keys())
    for i, row in results_df.iterrows():
        param_str = ', '.join([f"{param}={row[param]}" for param in param_columns])
        print(f"Params: {param_str}")
        print(f"  Val Acc: {row['avg_val_acc']:.4f}, Test Acc: {row['avg_test_acc']:.4f}")
    
    return results_df