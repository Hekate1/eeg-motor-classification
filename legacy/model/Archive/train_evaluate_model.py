#!/usr/bin/env python

"""
Main script for training and evaluating the Hybrid CNN/Transformer model.

This script provides command-line interface for:
1. Cross-validation
2. Pretraining
3. Fine-tuning
4. Evaluation
5. Subject splitting (creates training/testing sets)

Example usage:
    # Cross-validation on a single subject
    python train_evaluate_model.py cv --subject 001 --output_dir results/cv

    # Pretrain on multiple subjects
    python train_evaluate_model.py pretrain --subjects 001 002 003 004 --output_dir results/pretrained

    # Fine-tune for a specific subject
    python train_evaluate_model.py finetune --subject 009 --model results/pretrained/pretrained_model.pt --output_dir results/finetuned

    # Evaluate model on multiple subjects
    python train_evaluate_model.py evaluate --subjects 001 002 003 --model results/finetuned/finetuned_model.pt --output_dir results/evaluation
    
    # Split subjects into training and testing sets
    python train_evaluate_model.py split_subjects --total 90 --test 10 --output_dir subject_lists
    
Note: Processed data should be available in the 'processed_data' directory.
      Run enhanced_preprocessing.py first to process the raw EEG data.
"""

import os
import argparse
import numpy as np
import time
import datetime
import random
import multiprocessing

# Set environment variables for multi-threading before importing torch
os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(multiprocessing.cpu_count())
os.environ["NUMEXPR_NUM_THREADS"] = str(multiprocessing.cpu_count())
os.environ["OPENBLAS_NUM_THREADS"] = str(multiprocessing.cpu_count()) 

import torch

from models.evaluation import (
    load_and_prepare_eeg_data,
    cross_validate_model,
    pretrain_model,
    finetune_subject_specific,
    evaluate_model_on_subjects,
    plot_confusion_matrices
)


def create_subject_split(total_subjects=90, test_subjects=10, output_dir=None, random_seed=42):
    """
    Create a random split of subjects into training and testing sets.
    
    Args:
        total_subjects: Total number of subjects available
        test_subjects: Number of subjects to use for testing
        output_dir: Directory to save the subject list files
        random_seed: Random seed for reproducibility
    
    Returns:
        train_subjects: List of subject IDs for training
        test_subjects: List of subject IDs for testing
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Create a list of all available subject IDs with leading zeros (e.g., "001", "002", etc.)
    all_subjects = [f"{i:03d}" for i in range(1, total_subjects + 1)]
    
    # Randomly shuffle subjects
    random.shuffle(all_subjects)
    
    # Split into training and testing sets
    test_subject_list = all_subjects[:test_subjects]
    train_subject_list = all_subjects[test_subjects:]
    
    # Sort for better readability
    test_subject_list.sort()
    train_subject_list.sort()
    
    print(f"Created split: {len(train_subject_list)} subjects for training, {len(test_subject_list)} subjects for testing")
    
    # Save to files if output directory is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save training subjects
        train_file = os.path.join(output_dir, "train_subjects.txt")
        with open(train_file, 'w') as f:
            for subject in train_subject_list:
                f.write(f"{subject}\n")
        print(f"Training subjects saved to {train_file}")
        
        # Save testing subjects
        test_file = os.path.join(output_dir, "test_subjects.txt")
        with open(test_file, 'w') as f:
            for subject in test_subject_list:
                f.write(f"{subject}\n")
        print(f"Testing subjects saved to {test_file}")
    
    return train_subject_list, test_subject_list


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate Hybrid CNN/Transformer model")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Cross-validation command
    cv_parser = subparsers.add_parser('cv', help='Run cross-validation')
    cv_parser.add_argument('--subject', type=str, required=True, help='Subject ID')
    cv_parser.add_argument('--output_dir', type=str, help='Output directory')
    cv_parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for cross-validation')
    cv_parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs')
    cv_parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    cv_parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    cv_parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    cv_parser.add_argument('--n_layers', type=int, default=2, help='Number of transformer layers')
    cv_parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    cv_parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    cv_parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    cv_parser.add_argument('--n_csp', type=int, default=4, help='Number of CSP components')
    cv_parser.add_argument('--no_csp', action='store_true', help='Disable CSP features')
    cv_parser.add_argument('--no_freq', action='store_true', help='Disable frequency features')
    
    # Pretraining command
    pretrain_parser = subparsers.add_parser('pretrain', help='Pretrain model on multiple subjects')
    pretrain_group = pretrain_parser.add_mutually_exclusive_group(required=True)
    pretrain_group.add_argument('--subjects', type=str, nargs='+', help='Subject IDs to use for pretraining')
    pretrain_group.add_argument('--subjects_file', type=str, help='File containing subject IDs (one per line)')
    pretrain_parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    pretrain_parser.add_argument('--n_epochs', type=int, default=200, help='Number of epochs')
    pretrain_parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    pretrain_parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    pretrain_parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    pretrain_parser.add_argument('--n_layers', type=int, default=2, help='Number of transformer layers')
    pretrain_parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    pretrain_parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    pretrain_parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    pretrain_parser.add_argument('--n_csp', type=int, default=4, help='Number of CSP components')
    pretrain_parser.add_argument('--no_csp', action='store_true', help='Disable CSP features')
    pretrain_parser.add_argument('--no_freq', action='store_true', help='Disable frequency features')
    
    # Fine-tuning command
    finetune_parser = subparsers.add_parser('finetune', help='Fine-tune model for a specific subject')
    finetune_parser.add_argument('--subject', type=str, required=True, help='Subject ID')
    finetune_parser.add_argument('--model', type=str, required=True, help='Path to pretrained model')
    finetune_parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    finetune_parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs')
    finetune_parser.add_argument('--n_csp', type=int, default=4, help='Number of CSP components')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model on subjects')
    eval_group = eval_parser.add_mutually_exclusive_group(required=True)
    eval_group.add_argument('--subjects', type=str, nargs='+', help='Subject IDs to evaluate on')
    eval_group.add_argument('--subjects_file', type=str, help='File containing subject IDs (one per line)')
    eval_parser.add_argument('--model', type=str, required=True, help='Path to model')
    eval_parser.add_argument('--output_dir', type=str, help='Output directory')
    eval_parser.add_argument('--n_csp', type=int, default=4, help='Number of CSP components')
    
    # Subject splitting command
    split_parser = subparsers.add_parser('split_subjects', help='Create training and testing subject lists')
    split_parser.add_argument('--total', type=int, default=90, help='Total number of subjects available')
    split_parser.add_argument('--test', type=int, default=10, help='Number of subjects to use for testing')
    split_parser.add_argument('--output_dir', type=str, default='subject_lists', help='Directory to save subject lists')
    split_parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    return parser.parse_args()


def read_subjects_from_file(file_path):
    """Read subject IDs from a file, one per line"""
    with open(file_path, 'r') as f:
        # Strip whitespace and filter out empty lines
        subjects = [line.strip() for line in f.readlines() if line.strip()]
    return subjects


def main():
    # Enable PyTorch multi-threading
    import torch
    import multiprocessing
    
    # Get core count
    num_cores = multiprocessing.cpu_count()
    
    # Configure PyTorch threading
    torch.set_num_threads(num_cores)
    
    # Enable TF32 precision for better performance on newer CPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # Set benchmark mode to use the most optimized algorithms
    torch.backends.cudnn.benchmark = True
    
    print(f"PyTorch using {num_cores} CPU threads")
    print(f"Environment variables: OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'Not set')}")
    
    # Parse command-line arguments
    args = parse_args()
    
    # Set output directory
    if args.command in ['cv', 'evaluate'] and args.output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"results_{args.command}_{timestamp}"
        print(f"No output directory specified. Using: {args.output_dir}")
    
    # Create output directory if it doesn't exist
    if hasattr(args, 'output_dir') and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Process command
    if args.command == 'split_subjects':
        # Create subject split
        train_subjects, test_subjects = create_subject_split(
            total_subjects=args.total,
            test_subjects=args.test,
            output_dir=args.output_dir,
            random_seed=args.seed
        )
        print(f"Training subjects ({len(train_subjects)}): {', '.join(train_subjects[:5])}... (first 5)")
        print(f"Testing subjects ({len(test_subjects)}): {', '.join(test_subjects)}")
        
    elif args.command == 'cv':
        # Load data for the subject
        data = load_and_prepare_eeg_data(
            [args.subject],
            n_csp_components=args.n_csp
        )
        
        # Make sure the subject data is available
        if args.subject not in data:
            print(f"Data for subject {args.subject} not found. Exiting.")
            return
        
        # Run cross-validation
        model_params = {
            'batch_size': args.batch_size,
            'embedding_dim': args.embedding_dim,
            'n_heads': args.n_heads,
            'n_layers': args.n_layers,
            'dropout': args.dropout,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'use_csp': not args.no_csp,
            'use_freq': not args.no_freq,
            'n_csp_components': args.n_csp
        }
        
        results = cross_validate_model(
            data,
            args.subject,
            n_folds=args.n_folds,
            n_epochs=args.n_epochs,
            output_dir=args.output_dir,
            save_models=True,
            **model_params
        )
        
    elif args.command == 'pretrain':
        # Get subjects from file if provided
        if hasattr(args, 'subjects_file') and args.subjects_file:
            subjects = read_subjects_from_file(args.subjects_file)
            print(f"Loaded {len(subjects)} subjects from {args.subjects_file}")
            # Print the first few subjects to verify format
            print(f"First few subjects: {subjects[:5]}")
        else:
            subjects = args.subjects
            
        # Check if processed data directory exists
        if not os.path.exists("processed_data"):
            print("WARNING: 'processed_data' directory not found. Make sure to run enhanced_preprocessing.py first.")
        else:
            print(f"Found processed_data directory at {os.path.abspath('processed_data')}")
            
        # Load data for all subjects
        data = load_and_prepare_eeg_data(
            subjects,
            n_csp_components=args.n_csp
        )
        
        # Make sure at least one subject has data
        if not any(subject in data for subject in subjects):
            print("No data found for any of the specified subjects. Exiting.")
            return
        
        # Run pretraining
        model_params = {
            'batch_size': args.batch_size,
            'embedding_dim': args.embedding_dim,
            'n_heads': args.n_heads,
            'n_layers': args.n_layers,
            'dropout': args.dropout,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'use_csp': not args.no_csp,
            'use_freq': not args.no_freq,
            'n_csp_components': args.n_csp
        }
        
        model = pretrain_model(
            data,
            subjects,
            args.output_dir,
            n_epochs=args.n_epochs,
            **model_params
        )
        
    elif args.command == 'finetune':
        # Load data for the subject
        data = load_and_prepare_eeg_data(
            [args.subject],
            n_csp_components=args.n_csp
        )
        
        # Make sure the subject data is available
        if args.subject not in data:
            print(f"Data for subject {args.subject} not found. Exiting.")
            return
        
        # Run fine-tuning
        results = finetune_subject_specific(
            args.model,
            data,
            args.subject,
            args.output_dir,
            n_epochs=args.n_epochs,
            n_csp_components=args.n_csp
        )
        
    elif args.command == 'evaluate':
        # Get subjects from file if provided
        if hasattr(args, 'subjects_file') and args.subjects_file:
            subjects = read_subjects_from_file(args.subjects_file)
            print(f"Loaded {len(subjects)} subjects from {args.subjects_file}")
        else:
            subjects = args.subjects
            
        # Load data for all subjects
        data = load_and_prepare_eeg_data(
            subjects,
            n_csp_components=args.n_csp
        )
        
        # Make sure at least one subject has data
        if not any(subject in data for subject in subjects):
            print("No data found for any of the specified subjects. Exiting.")
            return
        
        # Evaluate model
        results = evaluate_model_on_subjects(
            args.model,
            data,
            subjects,
            args.output_dir
        )
        
        # Plot confusion matrices
        plot_confusion_matrices(results, args.output_dir)
        
    else:
        print("No command specified. Use one of: cv, pretrain, finetune, evaluate, split_subjects")


if __name__ == "__main__":
    main() 