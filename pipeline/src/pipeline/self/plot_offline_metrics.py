#!/usr/bin/env python3
"""
Quick visualizer for offline metrics logged by train_offline_model.py.

Usage:
    python plot_offline_metrics.py --metrics results/offline_metrics.csv --out-dir results
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description="Plot offline training metrics.")
    ap.add_argument("--metrics", type=Path, default=Path("results/offline_metrics.csv"),
                    help="CSV produced by train_offline_model.py")
    ap.add_argument("--out-dir", type=Path, default=Path("results"),
                    help="Directory to save plots/summary")
    args = ap.parse_args()

    if not args.metrics.exists():
        raise SystemExit(f"Metrics file not found: {args.metrics}")

    df = pd.read_csv(args.metrics)
    if df.empty:
        raise SystemExit("Metrics file is empty.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Basic summary by mode (scratch vs transfer vs simple_* if present)
    summary = (
        df.groupby("mode")
        .agg(
            count=("val_accuracy", "size"),
            mean_val_acc=("val_accuracy", "mean"),
            std_val_acc=("val_accuracy", "std"),
        )
        .reset_index()
        .sort_values("mean_val_acc", ascending=False)
    )

    summary_path = args.out_dir / "offline_metrics_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote summary to {summary_path}")

    # Bar plot of individual runs
    plt.figure(figsize=(6, 4))
    colors = {"scratch": "#4c78a8", "transfer": "#f58518"}
    df_sorted = df.sort_values("val_accuracy", ascending=True)
    plt.barh(
        range(len(df_sorted)),
        df_sorted["val_accuracy"],
        color=[colors.get(m, "#72b7b2") for m in df_sorted["mode"]],
    )
    plt.yticks(range(len(df_sorted)), [f"{m} ({i})" for i, m in enumerate(df_sorted["mode"])] )
    plt.xlabel("Val accuracy")
    plt.title("Offline runs (val accuracy)")
    plt.tight_layout()
    plot_path = args.out_dir / "offline_metrics_runs.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Wrote per-run plot to {plot_path}")

    # Bar plot of summary means
    plt.figure(figsize=(5, 4))
    plt.bar(summary["mode"], summary["mean_val_acc"], yerr=summary["std_val_acc"], color="#4c78a8", alpha=0.8)
    plt.ylabel("Mean val accuracy")
    plt.title("Offline modes summary")
    plt.tight_layout()
    plot_path2 = args.out_dir / "offline_metrics_summary.png"
    plt.savefig(plot_path2, dpi=200)
    plt.close()
    print(f"Wrote summary plot to {plot_path2}")


if __name__ == "__main__":
    main()
