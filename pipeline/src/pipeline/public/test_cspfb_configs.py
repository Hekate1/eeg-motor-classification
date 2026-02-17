#!/usr/bin/env python3
"""Script to test different filter-bank CSP (CSPFB) configurations with LDA classification across cross-validation folds."""

import argparse
import numpy as np
import pandas as pd
import mne
from tqdm import tqdm
from data_utils import load_subject_data
from feature_utils import CSPTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
import sys
import gc
import ast


def generate_band_configs(start_freq, end_freq, window_sizes, steps):
    """
    Generate a dictionary of band configurations.  Keys are names like 'w4_s2' and values are lists of (low, high) tuples.
    """
    configs = {}
    for w in window_sizes:
        for s in steps:
            # Skip zero or negative step sizes to avoid infinite loops
            if s <= 0:
                print(f"Skipping invalid step size: {s}", file=sys.stderr)
                continue
            bands = []
            f = start_freq
            while f + w <= end_freq:
                bands.append((f, f + w))
                f += s
            if not bands:
                continue
            name = f"w{w}_s{s}"
            configs[name] = bands
    # baseline single band covering full range
    configs[f"full_{start_freq}_{end_freq}"] = [(start_freq, end_freq)]
    return configs


def load_top_configs(csv_path, top_n):
    """
    Load the top N configurations from a saved CSV file.
    
    Args:
        csv_path: Path to the CSV file containing previous results
        top_n: Number of top configurations to load
        
    Returns:
        List of configuration dictionaries
    """
    try:
        df = pd.read_csv(csv_path)
        # Convert string representation of bands back to list of tuples
        df['bands'] = df['bands'].apply(ast.literal_eval)
        
        # Convert bands to string representation for groupby
        df['bands_str'] = df['bands'].apply(str)
        
        # Get unique configurations and their average performance
        config_summary = df.groupby(['start_freq', 'end_freq', 'csp_components', 'config', 'bands_str']).agg({
            'mean_accuracy': 'mean'
        }).reset_index()
        
        # Sort by mean accuracy and get top N
        top_configs = config_summary.sort_values('mean_accuracy', ascending=False).head(top_n)
        
        # Convert to list of configuration dictionaries
        config_list = []
        for _, row in top_configs.iterrows():
            # Convert string representation back to list of tuples
            bands = ast.literal_eval(row['bands_str'])
            config_list.append({
                'start_freq': row['start_freq'],
                'end_freq': row['end_freq'],
                'csp_components': row['csp_components'],
                'config': row['config'],
                'bands': tuple(bands)
            })
        
        print(f"Loaded top {len(config_list)} configurations from {csv_path}")
        return config_list
    except Exception as e:
        print(f"Error loading configurations from {csv_path}: {e}", file=sys.stderr)
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Test CSP filter-bank configurations with a simple LDA classifier"
    )
    parser.add_argument(
        '--subject', type=str, default='001',
        help='Subject ID (e.g., 001)'
    )
    parser.add_argument(
        '--runs', nargs='+', default=['3','7','11'],
        help='Run IDs to load (e.g., 3 7 11)'
    )
    parser.add_argument(
        '--window-sizes', nargs='+', type=int, default=[2, 4, 6],
        help='List of filter-bank window sizes in Hz'
    )
    parser.add_argument(
        '--steps', nargs='+', type=int, default=[2,4],
        help='List of filter-bank step sizes (overlap) in Hz'
    )
    parser.add_argument(
        '--csp-components', nargs='+', type=int, default=[4, 8],
        help='List of CSP component counts per band to test'
    )
    parser.add_argument(
        '--start-freqs', nargs='+', type=float, default=[4.0, 8.0],
        help='List of start frequencies for band definitions'
    )
    parser.add_argument(
        '--end-freqs', nargs='+', type=float, default=[30.0, 38.0],
        help='List of end frequencies for band definitions'
    )
    parser.add_argument(
        '--n-folds', type=int, default=5,
        help='Number of cross-validation folds'
    )
    parser.add_argument(
        '--sfreq', type=float, default=160.0,
        help='Sampling frequency of the data'
    )
    parser.add_argument(
        '--output-csv', type=str, default=None,
        help='Optional CSV file to save results'
    )
    parser.add_argument(
        '--n-jobs', type=int, default=1,
        help='Number of parallel jobs for cross-validation (default=1)'
    )
    parser.add_argument(
        '--subjects', nargs='+', type=str, default=None,
        help='List of subject IDs to evaluate (default=None)'
    )
    parser.add_argument(
        '--load-configs', type=str, default=None,
        help='Path to CSV file containing previous results to load top configurations from'
    )
    parser.add_argument(
        '--top-n', type=int, default=5,
        help='Number of top configurations to load and test (default=5)'
    )
    args = parser.parse_args()

    # Set MNE log level to warning to reduce verbosity
    mne.set_log_level('ERROR')

    # Determine subject list; override single subject if --subjects provided
    subjects = args.subjects if hasattr(args, 'subjects') and args.subjects else [args.subject]
    
    # Load configurations either from CSV or generate new ones
    if args.load_configs:
        config_list = load_top_configs(args.load_configs, args.top_n)
        if not config_list:
            print("Failed to load configurations, exiting.", file=sys.stderr)
            return
    else:
        # Build full list of configurations
        config_list = []
        for sf in args.start_freqs:
            for ef in args.end_freqs:
                if sf >= ef:
                    print(f"Skipping invalid freq range: start={sf} >= end={ef}", file=sys.stderr)
                    continue
                try:
                    band_configs = generate_band_configs(sf, ef, args.window_sizes, args.steps)
                except Exception as e:
                    print(f"Skipping invalid band range [{sf}, {ef}]: {e}", file=sys.stderr)
                    continue
                for name, bands in band_configs.items():
                    for n_comp in args.csp_components:
                        config_list.append({
                            'start_freq': sf,
                            'end_freq': ef,
                            'csp_components': n_comp,
                            'config': name,
                            'bands': tuple(bands)
                        })

    print(f"Testing {len(config_list)} configurations over {len(subjects)} subjects...")
    
    # Evaluate each configuration for each subject
    results = []
    for subj in tqdm(subjects, desc='Subjects'):
        print(f"\nSubject: {subj}")
        X_sub, y_sub, _ = load_subject_data(subject_id=subj, run_ids=args.runs, use_motor_channels=True)
        for cfg in tqdm(config_list, desc=f'Configurations for {subj}', leave=False):
            sf, ef = cfg['start_freq'], cfg['end_freq']
            n_comp, name, bands = cfg['csp_components'], cfg['config'], list(cfg['bands'])
            scores = []
            for train_idx, test_idx in StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42).split(X_sub, y_sub):
                try:
                    csp_fold = CSPTransformer(X_sub[train_idx], y_sub[train_idx], per_subject=False,
                                              n_components=n_comp, filter_bank=True, bands=bands, sfreq=args.sfreq)
                    X_tr = csp_fold.transform(X_sub[train_idx])
                    X_te = csp_fold.transform(X_sub[test_idx])
                    lda_f = LinearDiscriminantAnalysis()
                    lda_f.fit(X_tr, y_sub[train_idx])
                    scores.append(lda_f.score(X_te, y_sub[test_idx]))
                except Exception as e:
                    print(f"Skipping fold for subj={subj}, cfg={name}, comp={n_comp}: {e}", file=sys.stderr)
                finally:
                    for obj in (csp_fold, X_tr, X_te, lda_f):
                        del obj
                    gc.collect()
            if not scores:
                continue
            results.append({
                'subject': subj,
                'start_freq': sf,
                'end_freq': ef,
                'csp_components': n_comp,
                'config': name,
                'bands': bands,
                'mean_accuracy': np.mean(scores)
            })
    df = pd.DataFrame(results)
    # Order results by descending mean accuracy
    # Aggregate performance across subjects, excluding the 'subject' column from mean calculation
    summary = df.groupby(['start_freq','end_freq','csp_components','config']).agg({
        'mean_accuracy': 'mean',
        'subject': lambda x: list(x)  # Keep track of which subjects were used
    }).reset_index()
    summary = summary.sort_values(by='mean_accuracy', ascending=False).reset_index(drop=True)
    print("\nOverall average performance across subjects:")
    for idx, row in summary.iterrows():
        print(f"{idx+1}. {row['config']} [{row['start_freq']}-{row['end_freq']} Hz] with {row['csp_components']} components: {row['mean_accuracy']:.4f}")
    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"Results saved to {args.output_csv}")


if __name__ == '__main__':
    main()
