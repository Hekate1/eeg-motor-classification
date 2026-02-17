#!/usr/bin/env python3
"""
Experiment-side training script: preprocess raw .fif data and train a HybridModelClassifier.
"""

import os
import sys
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import mne
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# import epoch_data from online pipeline
# Force headless mode for pyglet/psychopy when no display is available
os.environ.setdefault("PYGLET_HEADLESS", "True")
from headset_frontend.online_mi_pipeline import epoch_data, EPOCH_TMIN, EPOCH_TMAX
from headset_frontend.online_preprocessing import preprocess_epoch_data

# Import model-side classes from the public-dataset pipeline package.
#
# NOTE: `pipeline.public` is originally written as a script-style module set.
# We keep imports stable by temporarily adding that directory to sys.path.
public_src_path = Path(__file__).resolve().parents[1] / "public"
sys.path.insert(0, str(public_src_path))
from hybrid_cnn_transformer import HybridModelClassifier  # type: ignore
from feature_modules import CSPModule  # type: ignore
from mne.decoding import CSP


def _as_run_str_list(runs: Optional[Sequence[Union[str, int]]]) -> List[str]:
    """Normalize run IDs to zero-padded strings like ['01','02',...]."""
    if runs is None:
        return []
    out: List[str] = []
    for r in runs:
        if isinstance(r, int):
            out.append(f"{r:02d}")
        else:
            s = str(r).strip()
            if s.isdigit():
                out.append(f"{int(s):02d}")
            else:
                out.append(s)
    return out


def _kv_notes(parts: Dict[str, Any]) -> str:
    """Encode metadata dict into stable ';' separated notes string."""
    def _fmt(v: Any) -> str:
        if isinstance(v, (list, tuple)):
            # repr keeps quotes so notebook regex parsing stays robust
            return repr([str(x) for x in v])
        return str(v)

    return ";".join([f"{k}={_fmt(v)}" for k, v in parts.items() if v is not None and v != ""])


def _load_raw_runs(
    data_dir: Path,
    subject: str,
    runs: Sequence[Union[str, int]],
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], float]:
    """Load and epoch a set of raw FIF runs for a single subject."""
    runs_s = _as_run_str_list(runs)
    raw_data: List[Tuple[np.ndarray, np.ndarray]] = []
    sfreq: Optional[float] = None
    for run in runs_s:
        path = data_dir / f"sub-{subject}_run-{int(run):02d}_online_raw.fif"
        if not path.exists():
            continue
        raw = mne.io.read_raw_fif(str(path), preload=True, verbose=False)
        sfreq = float(raw.info["sfreq"])
        events = mne.find_events(raw, stim_channel="STI 014", shortest_event=1)
        X_run, y_run = preprocess_epoch_data(raw, events, EPOCH_TMIN, EPOCH_TMAX)
        y_run = y_run - 1  # 1/2 -> 0/1
        raw_data.append((X_run, y_run))
    if not raw_data:
        raise ValueError(f"No epoch data loaded for subject={subject} runs={runs_s} in {data_dir}")
    if sfreq is None:
        raise RuntimeError("Failed to determine sampling rate (sfreq)")
    return raw_data, sfreq


def _concat_truncate(raw_data: List[Tuple[np.ndarray, np.ndarray]], common_length: int) -> Tuple[np.ndarray, np.ndarray]:
    X = np.concatenate([d[0][:, :, :common_length] for d in raw_data], axis=0)
    y = np.concatenate([d[1] for d in raw_data], axis=0)
    return X, y


def train_and_eval_on_runs(
    *,
    data_dir: Path,
    subject: str,
    train_runs: Sequence[Union[str, int]],
    test_runs: Sequence[Union[str, int]],
    metrics_csv: Path = Path("results/offline_metrics.csv"),
    output_dir: Path = Path("models"),
    model_name: str = "offline_model",
    base_model_path: Optional[Path] = None,
    seed: int = 42,
    n_epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    dropout: float = 0.1,
    embedding_dim: int = 16,
    n_heads: int = 2,
    n_layers: int = 3,
    n_csp_components: int = 6,
    freeze_backbone: bool = False,
    refit_normalizer: bool = True,
    split_tag: str = "explicit_runs",
    metric_name: str = "val_accuracy",
    notes_extra: str = "",
) -> float:
    """
    Train on an explicit list of runs and evaluate on held-out runs.

    This is the entrypoint used by `offline_analysis.ipynb` for run-wise splits
    (e.g., LORO / holdout). The held-out-run accuracy is logged into `val_accuracy`.
    """
    train_runs_s = _as_run_str_list(train_runs)
    test_runs_s = _as_run_str_list(test_runs)
    if not train_runs_s or not test_runs_s:
        raise ValueError(f"train_runs and test_runs must both be non-empty (got train={train_runs_s}, test={test_runs_s})")

    train_raw, sfreq_train = _load_raw_runs(data_dir, subject, train_runs_s)
    test_raw, sfreq_test = _load_raw_runs(data_dir, subject, test_runs_s)
    sfreq = sfreq_train if sfreq_train else sfreq_test

    # Determine common time dimension divisible by 32 across BOTH splits
    n_times_list = [d[0].shape[2] for d in (train_raw + test_raw)]
    min_n_times = min(n_times_list)
    common_length = (min_n_times // 32) * 32
    if common_length <= 0:
        raise ValueError(f"Common time length too small: {min_n_times}")

    X_train, y_train = _concat_truncate(train_raw, common_length)
    X_val, y_val = _concat_truncate(test_raw, common_length)
    idx_train = np.zeros(len(y_train), dtype=int)
    idx_val = np.zeros(len(y_val), dtype=int)
    subjects = [str(subject)]
    all_runs = sorted(set(train_runs_s + test_runs_s))

    # Build an args-like namespace for the existing training functions
    args = argparse.Namespace()
    args.data_dir = data_dir
    args.runs = all_runs
    args.train_runs = train_runs_s
    args.test_runs = test_runs_s
    args.val_split = 0.0  # explicit split; not used for train/val selection
    args.seed = int(seed)
    args.n_epochs = int(n_epochs)
    args.batch_size = int(batch_size)
    args.lr = float(lr)
    args.weight_decay = float(weight_decay)
    args.dropout = float(dropout)
    args.embedding_dim = int(embedding_dim)
    args.n_heads = int(n_heads)
    args.n_layers = int(n_layers)
    args.n_csp_components = int(n_csp_components)
    args.simple_model = None
    args.metrics_csv = Path(metrics_csv)
    args.output_dir = Path(output_dir)
    args.plots_dir = args.output_dir / "plots"
    args.model_name = str(model_name)
    args.base_model_path = Path(base_model_path) if base_model_path else None
    args.freeze_backbone = bool(freeze_backbone)
    args.refit_normalizer = bool(refit_normalizer)
    args.split = str(split_tag)
    args.metric = str(metric_name)
    args.notes_extra = str(notes_extra) if notes_extra else ""

    train_deep_model(
        args,
        X_train,
        y_train,
        idx_train,
        X_val,
        y_val,
        idx_val,
        subjects,
        all_runs,
        sfreq,
    )

    # Return the last-eval accuracy that was logged/surfaced
    # (train_deep_model prints it; here we recompute quickly from the saved model)
    model_path = args.output_dir / f"{args.model_name}.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.long, device=device)
    idx_val_t = torch.tensor(idx_val, dtype=torch.long, device=device)
    best_clf, _ = HybridModelClassifier.load_model(str(model_path), device=device)
    target_channels = getattr(best_clf.model, "n_channels", X_val_t.shape[1])
    if X_val_t.size(1) != target_channels:
        # match channels for transfer models
        if X_val_t.size(1) < target_channels:
            pad = torch.zeros(X_val_t.size(0), target_channels - X_val_t.size(1), X_val_t.size(2), device=device)
            X_val_t = torch.cat([X_val_t, pad], dim=1)
        else:
            X_val_t = X_val_t[:, :target_channels, :]
    eval_res = best_clf.evaluate(X_val_t, y_val_t, subject_indices=idx_val_t)
    return float(eval_res.get("accuracy"))


def _append_metrics_row(csv_path: Path, row: dict):
    """Append a single metrics row to a CSV, creating headers if missing."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _plot_history(history: dict, out_path: Path):
    """Save simple train/val accuracy & loss curves for quick inspection."""
    if not history:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    if "train_acc" in history:
        plt.plot(history["train_acc"], label="train_acc")
    if "val_acc" in history:
        plt.plot(history["val_acc"], label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.title("Training history (acc)")
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    if "train_loss" in history:
        plt.plot(history["train_loss"], label="train_loss")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Training history (loss)")
    plt.tight_layout()
    plt.savefig(out_path.with_name(out_path.stem + "_loss.png"), dpi=200)
    plt.close()


def train_simple_model(args, X_train, y_train, idx_train, X_val, y_val, idx_val):
    """Train a simple sklearn model using mne.decoding.CSP features."""
    # Compute CSP features directly
    csp = CSP(n_components=args.n_csp_components, log=True)
    csp.fit(X_train, y_train)
    X_feat_train = csp.transform(X_train)
    X_feat_val = csp.transform(X_val)
    # Choose model
    if args.simple_model == 'rf':
        clf = RandomForestClassifier()
    else:
        clf = SVC()
    # Train and evaluate
    clf.fit(X_feat_train, y_train)
    y_pred = clf.predict(X_feat_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"Validation accuracy for {args.simple_model}: {acc:.4f}")

    # Log basic metrics for classical baselines
    notes = _kv_notes(
        {
            "csp_components": args.n_csp_components,
            "runs": args.runs,
            "seed": getattr(args, "seed", None),
            "val_split": getattr(args, "val_split", None),
            "split": getattr(args, "split", None),
            "train_runs": getattr(args, "train_runs", None),
            "test_runs": getattr(args, "test_runs", None),
            "metric": getattr(args, "metric", None),
            "notes_extra": getattr(args, "notes_extra", None),
        }
    )
    _append_metrics_row(
        args.metrics_csv,
        {
            "timestamp": datetime.utcnow().isoformat(),
            "mode": f"simple_{args.simple_model}",
            "base_model": "",
            "subjects": len(np.unique(idx_train)),
            "trials": len(y_train) + len(y_val),
            "val_accuracy": acc,
            "notes": notes,
        },
    )


def train_deep_model(args, X_train, y_train, idx_train, X_val, y_val, idx_val, subjects, runs, sfreq):
    """Train deep HybridModelClassifier and save the model."""
    # Move to tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    idx_train_t = torch.tensor(idx_train, dtype=torch.long, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.long, device=device)
    idx_val_t = torch.tensor(idx_val, dtype=torch.long, device=device)

    def _match_channels(x: torch.Tensor, target_channels: int) -> torch.Tensor:
        """Pad or truncate channel dimension to match target_channels."""
        if x.size(1) == target_channels:
            return x
        if x.size(1) < target_channels:
            pad = torch.zeros(x.size(0), target_channels - x.size(1), x.size(2), device=x.device, dtype=x.dtype)
            return torch.cat([x, pad], dim=1)
        # truncate extra channels
        return x[:, :target_channels, :]

    # Build feature modules (always refit on current subject data)
    def _build_feature_modules(configs=None):
        modules = []
        if configs is None:
            configs = [
                {
                    "name": "CSPModule",
                    "params": {
                        "n_components": args.n_csp_components,
                        "per_subject": True,
                        "filter_bank": False,
                        "bands": None,
                        "sfreq": sfreq,
                    },
                }
            ]
        for cfg in configs:
            if cfg["name"] != "CSPModule":
                raise ValueError(f"Unsupported feature module type for offline training: {cfg['name']}")
            params = dict(cfg["params"])
            # Ensure sampling rate matches the current dataset
            params["sfreq"] = sfreq
            mod = CSPModule(**params)
            mod.fit(X_train_t, y_train_t, subject_indices=idx_train_t)
            modules.append(mod)
        return modules

    feature_modules = None
    clf = None
    metadata = {}

    if args.base_model_path:
        if not Path(args.base_model_path).exists():
            raise FileNotFoundError(f"Base model not found: {args.base_model_path}")
        print(f"Loading base model from {args.base_model_path}")
        clf, metadata = HybridModelClassifier.load_model(str(args.base_model_path))
        target_channels = getattr(clf.model, "n_channels", X_train_t.shape[1])
        X_train_t = _match_channels(X_train_t, target_channels)
        X_val_t = _match_channels(X_val_t, target_channels)
        # Rebuild feature modules based on saved config, but fit on self data
        fm_configs = metadata.get("feature_modules")
        feature_modules = _build_feature_modules(fm_configs)
        clf.feature_modules = feature_modules
        clf.use_feature_modules = True
        # Optionally refit normalizer on current data to reduce domain shift
        if args.refit_normalizer:
            clf.normalizer = None
        # Update training hyperparameters for fine-tuning
        clf.n_epochs = args.n_epochs
        clf.lr = args.lr
        clf.batch_size = args.batch_size
        clf.weight_decay = args.weight_decay
        clf.dropout = args.dropout
        if args.freeze_backbone:
            unfrozen = [layer for layer in ['fusion', 'feature_encoder', 'adapter_head', 'combiner_head'] if hasattr(clf.model, layer)]
            print(f"Freezing backbone; unfrozen layers: {unfrozen}")
            clf.freeze_layers(unfrozen_layers=unfrozen) if unfrozen else None
        model_mode = "transfer"
    else:
        target_channels = X_train_t.shape[1]
        X_train_t = _match_channels(X_train_t, target_channels)
        X_val_t = _match_channels(X_val_t, target_channels)
        feature_modules = _build_feature_modules()
        n_classes = len(np.unique(y_train))
        clf = HybridModelClassifier(
            n_classes=n_classes,
            embedding_dim=args.embedding_dim,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout=args.dropout,
            lr=args.lr,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            weight_decay=args.weight_decay,
            use_feature_modules=True,
            model_type='transformer'
        )
        clf.feature_modules = feature_modules
        model_mode = "scratch"

    results = clf.train_and_evaluate(
        X_train_t, y_train_t,
        X_val_t, y_val_t,
        train_subject_indices=idx_train_t,
        val_subject_indices=idx_val_t,
        early_stopping_patience=50,
        use_lr_scheduler='onecycle',
        fine_tuning=bool(args.base_model_path)
    )

    # Save model and compute margin stats
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / f"{args.model_name}.pt"
    # Match channels again in case transforms introduce device copies
    X_train_t = _match_channels(X_train_t, target_channels)
    X_val_t = _match_channels(X_val_t, target_channels)

    feat_train_list = [fm.transform(X_train_t, idx_train_t) for fm in feature_modules]
    X_feat_train = torch.cat(feat_train_list, dim=1)
    X_train_norm_all, X_feat_train_norm = clf.normalizer.transform(X_train_t, X_feat_train)
    with torch.no_grad():
        # Ensure features are on the same device as model parameters for margin calc
        model_device = next(clf.model.parameters()).device
        X_feat_train_norm = X_feat_train_norm.to(model_device, dtype=torch.float32)
        margins = torch.abs(X_feat_train_norm @ clf.model.lda_W.t() + clf.model.lda_b)
    margin_mean = margins.mean(dim=0, keepdim=True).cpu().numpy()
    margin_std = margins.std(dim=0, keepdim=True, unbiased=False).cpu().numpy()
    metadata = {
        'subjects': subjects,
        'runs': runs,
        'results': results,
        'margin_mean': margin_mean,
        'margin_std': margin_std,
        'source_model': str(args.base_model_path) if args.base_model_path else None,
        'mode': model_mode,
        'timestamp': datetime.utcnow().isoformat(),
    }
    clf.save_model(str(model_path), metadata=metadata)
    print(f"Model saved to {model_path}")

    # Persist history for later plotting/aggregation
    if isinstance(results, dict) and 'history' in results:
        history = results['history']
        history_path = args.output_dir / f"{args.model_name}_history.json"
        with history_path.open("w") as f:
            json.dump(history, f)
        _plot_history(history, args.plots_dir / f"{args.model_name}_history")

    # Print history metrics if available
    if isinstance(results, dict) and 'history' in results:
        hist = results['history']
        print("Training history metrics:")
        for key in ['train_acc', 'val_acc']:
            if key in hist:
                vals = hist[key]
                nan_indices = [i for i, v in enumerate(vals) if np.isnan(v)]
                if nan_indices:
                    print(f"  {key}: contains NaN at positions {nan_indices}")
                print(f"  {key}: {vals}")

    # Evaluate saved model
    print("Evaluating saved model on validation set")
    best_clf, _ = HybridModelClassifier.load_model(str(model_path))
    eval_res = best_clf.evaluate(X_val_t, y_val_t, subject_indices=idx_val_t)
    print(f"Final validation accuracy (saved model): {eval_res['accuracy']:.4f}")

    # Log consolidated metrics
    notes = _kv_notes(
        {
            "csp_components": args.n_csp_components,
            "runs": runs,
            "target_channels": target_channels,
            "seed": getattr(args, "seed", None),
            "val_split": getattr(args, "val_split", None),
            "split": getattr(args, "split", None),
            "train_runs": getattr(args, "train_runs", None),
            # Prefer a single held-out run tag if it's exactly one run
            "test_run": (args.test_runs[0] if getattr(args, "test_runs", None) and len(args.test_runs) == 1 else None),
            "test_runs": (getattr(args, "test_runs", None) if getattr(args, "test_runs", None) and len(args.test_runs) != 1 else None),
            "metric": getattr(args, "metric", None),
            "notes_extra": getattr(args, "notes_extra", None),
        }
    )
    _append_metrics_row(
        args.metrics_csv,
        {
            "timestamp": datetime.utcnow().isoformat(),
            "mode": model_mode,
            "base_model": str(args.base_model_path) if args.base_model_path else "",
            "subjects": len(subjects),
            "trials": len(y_train) + len(y_val),
            "val_accuracy": eval_res.get('accuracy'),
            "notes": notes,
        },
    )


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess raw data and train a pretrained HybridModelClassifier'
    )
    parser.add_argument('--data-dir', type=Path, default=Path('data/self'),
                        help='Directory containing sub-XX_run-YY_online_raw.fif files')
    parser.add_argument('--runs', nargs='+', default=None,
                        help='Run IDs to include (e.g. 01 02)')
    parser.add_argument('--train-runs', nargs='+', default=None,
                        help='Explicit train run IDs (overrides --val-split split)')
    parser.add_argument('--test-runs', nargs='+', default=None,
                        help='Explicit test run IDs (overrides --val-split split)')
    parser.add_argument('--split', type=str, default='random_stratified',
                        help="Split tag to log in metrics notes (e.g. random_stratified, leave_one_run_out)")
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Fraction of data to use for validation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--n-epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--simple-model', choices=['rf', 'svm'], default=None,
                        help='Train a simple sklearn model instead of deep model')
    parser.add_argument('--embedding-dim', type=int, default=16,
                        help='Model embedding dimension')
    parser.add_argument('--n-heads', type=int, default=2,
                        help='Number of transformer heads')
    parser.add_argument('--n-layers', type=int, default=3,
                        help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--n-csp-components', type=int, default=6,
                        help='Number of CSP components')
    parser.add_argument('--model-name', type=str, default='offline_model',
                        help='Filename prefix for saved model')
    parser.add_argument('--output-dir', type=Path, default=Path('runs/self/models'),
                        help='Directory to save trained model')
    parser.add_argument('--plots-dir', type=Path, default=None,
                        help='Directory to save training curves (default: <output-dir>/plots)')
    parser.add_argument('--metrics-csv', type=Path, default=Path('runs/self/results/offline_metrics.csv'),
                        help='CSV file to append run metrics for progress tracking')
    parser.add_argument('--base-model-path', type=Path, default=None,
                        help='Optional pretrained HybridModelClassifier to fine-tune (public dataset)')
    parser.add_argument('--freeze-backbone', action='store_true',
                        help='Freeze backbone layers during transfer; train only adapters/heads')
    parser.add_argument('--no-refit-normalizer', dest='refit_normalizer', action='store_false',
                        help='Keep the base model normalizer instead of refitting on self data')
    parser.set_defaults(refit_normalizer=True)
    args = parser.parse_args()
    if args.plots_dir is None:
        args.plots_dir = args.output_dir / "plots"

    # Discover all raw FIF files
    files = sorted(args.data_dir.glob('sub-*_run-*_online_raw.fif'))
    if not files:
        print(f"No .fif files found in {args.data_dir}")
        sys.exit(1)

    # Determine subjects and runs
    subjects = sorted({f.name.split('_')[0].split('-')[1] for f in files})
    if args.runs is None:
        runs = sorted({f.name.split('_')[1].split('-')[1] for f in files})
    else:
        runs = args.runs

    # Load and epoch data from all runs
    raw_data = []
    for si, subj in enumerate(subjects):
        for run in runs:
            path = args.data_dir / f'sub-{subj}_run-{int(run):02d}_online_raw.fif'
            print(f"Loading data from {path}")
            if not path.exists():
                continue
            raw = mne.io.read_raw_fif(str(path), preload=True, verbose=False)
            events = mne.find_events(raw, stim_channel='STI 014', shortest_event=1)
            X_run, y_run = preprocess_epoch_data(raw, events, EPOCH_TMIN, EPOCH_TMAX)
            # Convert labels from 1/2 to 0/1 for compatibility with classifier
            y_run = y_run - 1
            raw_data.append((X_run, y_run, np.full(len(y_run), si)))

    if not raw_data:
        print("No epoch data loaded; check your data directory and runs.")
        sys.exit(1)

    # Determine common time dimension divisible by 32
    n_times_list = [data[0].shape[2] for data in raw_data]
    min_n_times = min(n_times_list)
    common_length = (min_n_times // 32) * 32
    if common_length <= 0:
        raise ValueError(f"Common time length too small: {min_n_times}")
    print(f"Truncating all trials to {common_length} timepoints (divisible by 32)")

    # Truncate and concatenate data
    X = np.concatenate([d[0][:, :, :common_length] for d in raw_data], axis=0)
    y = np.concatenate([d[1] for d in raw_data], axis=0)
    subj_idx = np.concatenate([d[2] for d in raw_data], axis=0)
    print(f"Loaded data: {X.shape[0]} trials from {len(subjects)} subjects, time length {common_length}")

    # Train/validation split
    if args.train_runs and args.test_runs:
        # Explicit run split: build X/y for train and test separately (single subject case).
        # For multi-subject use-cases, prefer calling `train_and_eval_on_runs` from Python.
        subj = subjects[0]
        train_raw, sfreq_train = _load_raw_runs(args.data_dir, subj, args.train_runs)
        test_raw, sfreq_test = _load_raw_runs(args.data_dir, subj, args.test_runs)
        n_times_list2 = [d[0].shape[2] for d in (train_raw + test_raw)]
        min_n_times2 = min(n_times_list2)
        common_length2 = (min_n_times2 // 32) * 32
        X_train, y_train = _concat_truncate(train_raw, common_length2)
        X_val, y_val = _concat_truncate(test_raw, common_length2)
        idx_train = np.zeros(len(y_train), dtype=int)
        idx_val = np.zeros(len(y_val), dtype=int)
        args.val_split = 0.0
        args.train_runs = _as_run_str_list(args.train_runs)
        args.test_runs = _as_run_str_list(args.test_runs)
    else:
        X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
            X, y, subj_idx,
            test_size=args.val_split,
            stratify=y,
            random_state=args.seed
        )
        args.train_runs = None
        args.test_runs = None
    if args.simple_model:
        train_simple_model(args, X_train, y_train, idx_train, X_val, y_val, idx_val)
        sys.exit(0)

    train_deep_model(args, X_train, y_train, idx_train, X_val, y_val, idx_val, subjects, runs, raw.info['sfreq'])
    sys.exit(0)


if __name__ == '__main__':
    main() 