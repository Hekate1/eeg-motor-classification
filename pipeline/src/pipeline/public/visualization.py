import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import sys
import time

DEBUG = False

# Ensure plots directory exists
def save_plot(fig, filename, directory='plots'):
    """
    Save a matplotlib figure to the specified directory.
    
    Args:
        fig: Matplotlib figure object to save
        filename: Name of the file to save (without directory)
        directory: Directory to save to (default: 'plots')
        
    Returns:
        Path to the saved file
    """
    # Create plots directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Create full path
    filepath = os.path.join(directory, filename)
    
    # Save the figure
    fig.savefig(filepath)
    
    return filepath

def plot_base_model_results(results, n_subjects, save_fig=True, filename=None):
    """
    Plot training results for the base model.
    
    Args:
        results: Dictionary containing training history
        n_subjects: Number of subjects used in training
        save_fig: Whether to save the figure
        filename: Name of the file to save (without directory)
        
    Returns:
        Figure object
    """
    # Create figure with 2 subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot accuracy
    axs[0].plot(results['history']['train_acc'], label='Training')
    axs[0].plot(results['history']['val_acc'], label='Validation')
    axs[0].set_title(f'Model Accuracy ({n_subjects} Subjects)')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].grid(True)
    axs[0].legend()
    
    # Plot loss
    axs[1].plot(results['history']['train_loss'], label='Training')
    axs[1].plot(results['history']['val_loss'], label='Validation')
    axs[1].set_title(f'Model Loss ({n_subjects} Subjects)')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].grid(True)
    axs[1].legend()
    
    # Set plot title
    plt.suptitle(f'Base Model Training Results ({n_subjects} Subjects)', fontsize=16)
    plt.tight_layout()
    
    # Save figure if requested
    if save_fig:
        if filename is None:
            filename = f'base_model_{n_subjects}_subjects.png'
        save_plot(fig, filename)
    
    return fig


def plot_strategy_results(results, test_results, strategy, save_fig=True, filename=None):
    """
    Plot results from a specific augmentation strategy for a single subject.
    
    Args:
        results: Dictionary containing training history
        test_results: Dictionary containing test results
        strategy: Name of the augmentation strategy
        save_fig: Whether to save the figure
        filename: Name of the file to save (without directory)
        
    Returns:
        Figure object
    """
    # Create a figure with 2 subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot accuracy
    axs[0].plot(results['history']['train_acc'], label='Training')
    axs[0].plot(results['history']['val_acc'], label='Validation')
    axs[0].set_title(f'Model Accuracy - {strategy}')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].grid(True)
    axs[0].legend()
    
    # Plot loss
    axs[1].plot(results['history']['train_loss'], label='Training')
    axs[1].plot(results['history']['val_loss'], label='Validation')
    axs[1].set_title(f'Model Loss - {strategy}')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].grid(True)
    axs[1].legend()
    
    # Set plot title
    plt.suptitle(f'Fine-tuning Results - {strategy}', fontsize=16)
    plt.tight_layout()
    
    # Save figure if requested
    if save_fig:
        if filename is None:
            filename = f'strategy_{strategy.replace(" ", "_")}.png'
        save_plot(fig, filename)
    
    return fig


def plot_strategy_comparison(results, subject_id, save_fig=True, filename=None):
    """
    Plot comparison of different augmentation strategies for a single subject.
    
    Args:
        results: Dictionary with results for each strategy
        subject_id: ID of the subject
        save_fig: Whether to save the figure
        filename: Name of the file to save (without directory)
        
    Returns:
        Figure object
    """
    # Create figure
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    
    # Sort strategies by validation accuracy
    sorted_indices = np.argsort(results['accuracy'])[::-1]  # descending order
    sorted_strategies = [results['strategies'][i] for i in sorted_indices]
    sorted_accs = [results['accuracy'][i] for i in sorted_indices]
    sorted_test_accs = [results['test_accuracy'][i] for i in sorted_indices]
    
    # Plot validation accuracy
    axs[0].bar(range(len(sorted_strategies)), sorted_accs, color='skyblue')
    axs[0].set_title(f'Validation Accuracy by Strategy (Subject {subject_id})')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xticks(range(len(sorted_strategies)))
    axs[0].set_xticklabels(sorted_strategies, rotation=45, ha='right')
    axs[0].grid(True, axis='y', linestyle='--', alpha=0.7)
    axs[0].set_ylim(0, 1.0)
    for i, v in enumerate(sorted_accs):
        axs[0].text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    # Plot test accuracy
    axs[1].bar(range(len(sorted_strategies)), sorted_test_accs, color='salmon')
    axs[1].set_title(f'Test Accuracy by Strategy (Subject {subject_id})')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_xticks(range(len(sorted_strategies)))
    axs[1].set_xticklabels(sorted_strategies, rotation=45, ha='right')
    axs[1].grid(True, axis='y', linestyle='--', alpha=0.7)
    axs[1].set_ylim(0, 1.0)
    for i, v in enumerate(sorted_test_accs):
        axs[1].text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    plt.suptitle(f'Strategy Comparison for Subject {subject_id}', fontsize=16)
    plt.tight_layout()
    
    # Save figure if requested
    if save_fig:
        if filename is None:
            filename = f'subject_{subject_id}_strategy_comparison.png'
        save_plot(fig, filename)
    
    return fig


def visualize_cross_subject_results(fine_tune_results, save_fig=True, filename=None, save_csv=True, csv_filename=None):
    """
    Create a heatmap of best strategies for each subject.
    
    Args:
        fine_tune_results: Dictionary with results for each subject and strategy
        save_fig: Whether to save the figure
        filename: Name of the file to save (without directory)
        save_csv: Whether to save results to CSV
        csv_filename: Name of the CSV file to save (without directory)
        
    Returns:
        Figure object and pandas DataFrame with results
    """
    # Extract subject IDs and strategy names
    subject_ids = sorted(list(fine_tune_results.keys()))
    
    # Extract all unique strategies across all subjects
    all_strategies = []
    for subject_results in fine_tune_results.values():
        all_strategies.extend(subject_results['strategies'])
    strategies = sorted(list(set(all_strategies)))
    
    # Create empty DataFrame
    df = pd.DataFrame(index=subject_ids, columns=strategies)
    
    # Fill DataFrame with validation accuracies
    for subject_id in subject_ids:
        subject_results = fine_tune_results[subject_id]
        for i, strategy in enumerate(subject_results['strategies']):
            df.loc[subject_id, strategy] = subject_results['accuracy'][i]
    
    # Create a DataFrame to hold the best strategy for each subject
    best_strategies_df = pd.DataFrame(index=subject_ids, 
                                    columns=['best_strategy', 'val_accuracy', 'test_accuracy'])
    
    # Fill with best strategy and accuracy
    for subject_id in subject_ids:
        subject_results = fine_tune_results[subject_id]
        best_idx = np.argmax(subject_results['accuracy'])
        best_strategy = subject_results['strategies'][best_idx]
        best_val_acc = subject_results['accuracy'][best_idx]
        best_test_acc = subject_results['test_accuracy'][best_idx]
        
        best_strategies_df.loc[subject_id] = [best_strategy, best_val_acc, best_test_acc]
    
    # Calculate statistics for each strategy
    strategy_stats = pd.DataFrame(index=strategies,
                                 columns=['mean_val_acc', 'std_val_acc', 'mean_test_acc', 'std_test_acc', 'count'])
    
    # Generate relative performance DataFrame
    rel_df = pd.DataFrame(index=subject_ids, columns=strategies)
    
    # Fill statistics and relative performance
    for strategy in strategies:
        val_accs = []
        test_accs = []
        
        for subject_id in subject_ids:
            subject_results = fine_tune_results[subject_id]
            if strategy in subject_results['strategies']:
                idx = subject_results['strategies'].index(strategy)
                val_acc = subject_results['accuracy'][idx]
                test_acc = subject_results['test_accuracy'][idx]
                val_accs.append(val_acc)
                test_accs.append(test_acc)
                
                # Calculate relative performance compared to best strategy for this subject
                best_idx = np.argmax(subject_results['accuracy'])
                best_val_acc = subject_results['accuracy'][best_idx]
                rel_df.loc[subject_id, strategy] = val_acc / best_val_acc if best_val_acc > 0 else 0
        
        if val_accs:
            strategy_stats.loc[strategy, 'mean_val_acc'] = np.mean(val_accs)
            strategy_stats.loc[strategy, 'std_val_acc'] = np.std(val_accs)
            strategy_stats.loc[strategy, 'mean_test_acc'] = np.mean(test_accs)
            strategy_stats.loc[strategy, 'std_test_acc'] = np.std(test_accs)
            strategy_stats.loc[strategy, 'count'] = len(val_accs)
        else:
            strategy_stats.loc[strategy, 'count'] = 0

    # Ensure data is numeric by converting to float and filling NaNs
    df_numeric = df.astype(float)
    
    # Create heatmap of validation accuracies
    plt.figure(figsize=(12, 8))
    fig_val = plt.figure(figsize=(12, 8))
    
    # Create heatmap with numeric data
    sns.heatmap(df_numeric, annot=True, cmap='YlGnBu', cbar_kws={'label': 'Validation Accuracy'}, 
                fmt='.3f', linewidths=.5)
    
    plt.title('Strategy Performance Across Subjects (Validation Accuracy)', fontsize=16)
    plt.tight_layout()
    
    # Save figure if requested
    if save_fig:
        if filename is None:
            filename = 'cross_subject_strategy_heatmap.png'
        save_plot(fig_val, filename)
    
    # Create a DataFrame for test accuracies
    test_df = pd.DataFrame(index=subject_ids, columns=strategies)
    
    # Fill DataFrame with test accuracies
    for subject_id in subject_ids:
        subject_results = fine_tune_results[subject_id]
        for i, strategy in enumerate(subject_results['strategies']):
            test_df.loc[subject_id, strategy] = subject_results['test_accuracy'][i]
    
    # Ensure test data is numeric
    test_df_numeric = test_df.astype(float)
    
    # Create heatmap of test accuracies
    plt.figure(figsize=(12, 8))
    fig_test = plt.figure(figsize=(12, 8))
    
    # Create heatmap with test data
    sns.heatmap(test_df_numeric, annot=True, cmap='YlOrRd', cbar_kws={'label': 'Test Accuracy'}, 
                fmt='.3f', linewidths=.5)
    
    plt.title('Strategy Performance Across Subjects (Test Accuracy)', fontsize=16)
    plt.tight_layout()
    
    # Save test accuracy heatmap
    if save_fig:
        test_filename = 'cross_subject_test_strategy_heatmap.png'
        save_plot(fig_test, test_filename)
        print(f'Test accuracy heatmap saved to plots/{test_filename}')
    
    # Save results to CSV if requested
    if save_csv:
        if csv_filename is None:
            csv_filename = 'cross_subject_results.csv'
        
        # Create directory if it doesn't exist
        os.makedirs('plots/csvs', exist_ok=True)
        csv_path = os.path.join('plots/csvs', csv_filename)
        
        # Save to CSV
        best_strategies_df.to_csv(csv_path)
        print(f'Results saved to {csv_path}')
    
    # Print detailed statistics for all strategies
    print("\n" + "="*80)
    print("CROSS-SUBJECT STRATEGY PERFORMANCE SUMMARY")
    print("="*80)
    
    # Sort strategies by mean test accuracy in descending order
    sorted_stats = strategy_stats.sort_values(by='mean_test_acc', ascending=False)
    
    # Format and print the results
    print(f"{'Strategy':<25} {'Val Acc (mean ± std)':<25} {'Test Acc (mean ± std)':<25} {'Count':<6}")
    print("-"*80)
    
    for strategy, row in sorted_stats.iterrows():
        val_str = f"{row['mean_val_acc']:.4f} ± {row['std_val_acc']:.4f}"
        test_str = f"{row['mean_test_acc']:.4f} ± {row['std_test_acc']:.4f}"
        count = int(row['count'])
        print(f"{strategy:<25} {val_str:<25} {test_str:<25} {count:<6}")
    
    print("\nNote: Results sorted by mean test accuracy")
    print("="*80 + "\n")
    
    return df_numeric, strategy_stats, rel_df

def single_training_curve(fold_results, subject_id, strategy_name, fold_idx, save_fig=True):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(fold_results['history']['train_acc']) + 1)
    plt.plot(epochs, fold_results['history']['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(epochs, fold_results['history']['val_acc'], 'r-', label='Validation Accuracy')
    plt.plot(epochs, fold_results['history']['test_acc'], 'g-', label='Test Accuracy')
    
    # Add horizontal line at final accuracies
    final_train_acc = fold_results['history']['train_acc'][-1]
    final_val_acc = fold_results['history']['val_acc'][-1]
    final_test_acc = fold_results['history']['test_acc'][-1]
    
    plt.axhline(y=final_train_acc, color='b', linestyle='--', alpha=0.5)
    plt.axhline(y=final_val_acc, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=final_test_acc, color='g', linestyle='--', alpha=0.5)
    
    # Add text annotations for final values
    plt.text(len(epochs), final_train_acc, f'{final_train_acc:.4f}', 
            verticalalignment='bottom', horizontalalignment='right', color='blue')
    plt.text(len(epochs), final_val_acc, f'{final_val_acc:.4f}', 
            verticalalignment='bottom', horizontalalignment='right', color='red')
    plt.text(len(epochs), final_test_acc, f'{final_test_acc:.4f}', 
            verticalalignment='bottom', horizontalalignment='right', color='green')
    
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Subject {subject_id}, {strategy_name} - Fold {fold_idx+1} Accuracy Curves')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Debug printout when DEBUG is True
    if DEBUG:
        print(f"\n--- SINGLE FOLD TRAINING CURVE (Subject {subject_id}, {strategy_name}, Fold {fold_idx+1}) ---")
        print(f"{'Epoch':<6} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12}")
        print("-" * 45)
        for i, epoch in enumerate(epochs):
            train_acc = fold_results['history']['train_acc'][i]
            val_acc = fold_results['history']['val_acc'][i]
            test_acc = fold_results['history']['test_acc'][i]
            print(f"{epoch:<6} {train_acc:.4f}      {val_acc:.4f}      {test_acc:.4f}")
        print(f"\nFinal: Train={final_train_acc:.4f}, Val={final_val_acc:.4f}, Test={final_test_acc:.4f}")
        print("-" * 45)
    
    # Create directory for saving plots
    plots_dir = os.path.join(os.getcwd(), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    if save_fig:
        # Save the plot
        plot_path = os.path.join(plots_dir, f'subject_{subject_id}_{strategy_name.replace(" ", "_")}_fold{fold_idx+1}_accuracy.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

def average_training_curves(fold_histories, subject_id, strategy_name, save_fig=True):
        plt.figure(figsize=(10, 6))
        
        # Find the maximum number of epochs across all folds
        max_epochs = max([len(h['train_acc']) for h in fold_histories])
        epochs = range(1, max_epochs + 1)
        
        # Initialize arrays to store accuracies and counts
        avg_train_acc = np.zeros(max_epochs)
        avg_val_acc = np.zeros(max_epochs)
        avg_test_acc = np.zeros(max_epochs)
        count_train = np.zeros(max_epochs)
        count_val = np.zeros(max_epochs)
        count_test = np.zeros(max_epochs)
        
        # Calculate sum and count for each epoch
        for history in fold_histories:
            for i, (train_acc, val_acc) in enumerate(zip(history['train_acc'], history['val_acc'])):
                avg_train_acc[i] += train_acc
                avg_val_acc[i] += val_acc
                count_train[i] += 1
                count_val[i] += 1
            
            # Add test accuracy if available
            if 'test_acc' in history:
                for i, test_acc in enumerate(history['test_acc']):
                    avg_test_acc[i] += test_acc
                    count_test[i] += 1
        
        # Calculate averages (avoiding division by zero)
        avg_train_acc = np.divide(avg_train_acc, count_train, out=np.zeros_like(avg_train_acc), where=count_train!=0)
        avg_val_acc = np.divide(avg_val_acc, count_val, out=np.zeros_like(avg_val_acc), where=count_val!=0)
        avg_test_acc = np.divide(avg_test_acc, count_test, out=np.zeros_like(avg_test_acc), where=count_test!=0)
        
        # Determine if test accuracy exists across folds
        has_test_data = np.any(count_test > 0)
        
        # Plot individual fold curves for each fold with lighter opacity
        for history in fold_histories:
            fold_epochs = range(1, len(history['train_acc']) + 1)
            plt.plot(fold_epochs, history['train_acc'], color='blue', alpha=0.3)
            plt.plot(fold_epochs, history['val_acc'], color='red', alpha=0.3)
            if 'test_acc' in history:
                plt.plot(fold_epochs, history['test_acc'], color='green', alpha=0.3)

        # Plot the averages
        plt.plot(epochs, avg_train_acc, 'b-', label='Avg Train Accuracy')
        plt.plot(epochs, avg_val_acc, 'r-', label='Avg Validation Accuracy')
        if has_test_data:
            plt.plot(epochs, avg_test_acc, 'g-', label='Avg Test Accuracy')
        
        # Add final accuracy values as text
        last_valid_train = np.max(np.where(count_train > 0)[0]) if np.any(count_train > 0) else -1
        last_valid_val = np.max(np.where(count_val > 0)[0]) if np.any(count_val > 0) else -1
        last_valid_test = np.max(np.where(count_test > 0)[0]) if np.any(count_test > 0) else -1
        
        if last_valid_train >= 0:
            plt.text(epochs[last_valid_train], avg_train_acc[last_valid_train], 
                    f'{avg_train_acc[last_valid_train]:.4f}', 
                    verticalalignment='bottom', horizontalalignment='right', color='blue')
                    
        if last_valid_val >= 0:
            plt.text(epochs[last_valid_val], avg_val_acc[last_valid_val], 
                    f'{avg_val_acc[last_valid_val]:.4f}', 
                    verticalalignment='bottom', horizontalalignment='right', color='red')
                    
        if last_valid_test >= 0:
            plt.text(epochs[last_valid_test], avg_test_acc[last_valid_test], 
                    f'{avg_test_acc[last_valid_test]:.4f}', 
                    verticalalignment='bottom', horizontalalignment='right', color='green')
        
        # Add information about fold count to the plot
        fold_count_info = []
        for i in [max_epochs//4, max_epochs//2, 3*max_epochs//4, max_epochs-1]:
            if i < len(count_train):
                fold_count_info.append(f"Epoch {i+1}: {int(count_train[i])} folds")
        
        fold_text = "\n".join(fold_count_info)
        plt.figtext(0.02, 0.02, fold_text, fontsize=8)
        
        # Debug printout when DEBUG is True
        if DEBUG:
            print(f"\n--- AVERAGE TRAINING CURVES (Subject {subject_id}, {strategy_name}, {len(fold_histories)} Folds) ---")
            
            # Print header with columns
            header = f"{'Epoch':<6} {'Train Acc':<12} {'Val Acc':<12}"
            if has_test_data:
                header += f" {'Test Acc':<12}"
            header += f" {'#Folds':<8}"
            print(header)
            print("-" * (45 + (12 if has_test_data else 0)))
            
            # Print data for each epoch
            for i, epoch in enumerate(epochs):
                if count_train[i] > 0:  # Only print epochs with data
                    line = f"{epoch:<6} {avg_train_acc[i]:.4f}      {avg_val_acc[i]:.4f}     "
                    if has_test_data:
                        line += f" {avg_test_acc[i]:.4f}     " if count_test[i] > 0 else " ----        "
                    line += f" {int(count_train[i]):<8}"
                    print(line)
            
            # Print final values
            print("\nFinal Average Accuracies:")
            if last_valid_train >= 0:
                print(f"  Train: {avg_train_acc[last_valid_train]:.4f}")
            if last_valid_val >= 0:
                print(f"  Val  : {avg_val_acc[last_valid_val]:.4f}")
            if last_valid_test >= 0:
                print(f"  Test : {avg_test_acc[last_valid_test]:.4f}")
            print("-" * (45 + (12 if has_test_data else 0)))
        
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f'Subject {subject_id}, {strategy_name} - Average Accuracy Across {len(fold_histories)} Folds')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if save_fig:
            # Save the plot
            plots_dir = os.path.join(os.getcwd(), 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            plot_path = os.path.join(plots_dir, f'subject_{subject_id}_{strategy_name.replace(" ", "_")}_avg_accuracy.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()