#!/usr/bin/env python3
"""Generate the figures embedded in the README (docs/figures/).

Everything is computed from artifacts committed to the repo:
- runs/self/results/offline_metrics.csv  (training/eval metrics log)
- data/self/*.fif                        (self-collected runs)
- runs/self/models/*                     (saved model checkpoints)

Usage (from the repo root, with the [base,ml] extras installed):

    python scripts/generate_report_figures.py
"""
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "self"
RESULTS_DIR = REPO_ROOT / "runs" / "self" / "results"
MODELS_DIR = REPO_ROOT / "runs" / "self" / "models"
METRICS_CSV = RESULTS_DIR / "offline_metrics.csv"
OUT_DIR = REPO_ROOT / "docs" / "figures"

# Same module-path setup as the results notebook (works without pip install)
sys.path.insert(0, str(REPO_ROOT / "apps" / "headset_frontend" / "src"))
sys.path.insert(0, str(REPO_ROOT / "pipeline" / "src"))
sys.path.insert(0, str(REPO_ROOT / "pipeline" / "src" / "pipeline" / "public"))

# OpenBCI Cyton+Daisy default 10-20 electrode layout, in board channel order.
# The committed FIFs use generic names (C1..C16); this is the physical layout the
# headset was assembled with (same mapping as legacy/experiment/evaluate_mi_data.py).
ELECTRODES_1020 = [
    "Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2",
    "F7", "F8", "F3", "F4", "T7", "T8", "P3", "P4",
]

# ---------------------------------------------------------------- styling
INK = "#0b0b0b"        # primary text
INK_2 = "#52514e"      # secondary text
MUTED = "#898781"      # axis labels / ticks
GRID = "#e1e0d9"       # hairline gridlines
BASELINE = "#c3c2b7"   # axis spines
SERIES = {             # color follows the entity across every figure
    "transfer": "#2a78d6",  # blue
    "scratch": "#eb6834",   # orange
    "ts_lr": "#1baf7a",     # aqua
}
LABELS = {
    "transfer": "Transfer (pretrained, 90 subj.)",
    "scratch": "Deep net from scratch",
    "ts_lr": "Riemannian TS+LR",
}
CLASS_COLORS = {"Left": "#2a78d6", "Right": "#eb6834"}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.labelsize": 10,
    "axes.labelcolor": INK_2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.edgecolor": BASELINE,
    "legend.frameon": False,
})


def style_axis(ax, grid_axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def chance_line(ax, y=0.5, x=None, ha="left"):
    ax.axhline(y, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    xpos = x if x is not None else ax.get_xlim()[1]
    off = (2, 2) if ha == "left" else (-4, 2)
    ax.annotate("chance", (xpos, y), xytext=off, textcoords="offset points",
                color=MUTED, fontsize=8.5, ha=ha, va="bottom")


def save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------- metrics parsing

def parse_note(notes, key):
    m = re.search(rf"(?<![a-z_]){re.escape(key)}=([^;]+)", str(notes))
    return m.group(1).strip() if m else None


def load_metrics():
    df = pd.read_csv(METRICS_CSV)
    df["split"] = df["notes"].apply(lambda s: parse_note(s, "split") or "random_stratified")
    df["test_run"] = df["notes"].apply(lambda s: parse_note(s, "test_run"))
    df["recipe"] = df["notes"].apply(lambda s: parse_note(s, "recipe"))
    df["filter"] = df["notes"].apply(lambda s: parse_note(s, "filter"))
    runs = df["notes"].apply(
        lambda s: re.search(r"runs=\[(.*?)\]", str(s)).group(1) if re.search(r"runs=\[(.*?)\]", str(s)) else ""
    )
    df["n_runs"] = runs.apply(lambda s: len([x for x in s.split(",") if x.strip()]))
    return df


def v2_rows(df):
    """Leak-free calibration_curve_v2 rows, restricted to the headline arms:
    deep modes with the December hyperparameters on FIR-filtered data (the
    filter the pretrained base model was trained with), TS+LR with its own
    (unchanged) preprocessing. Deep rows average over 3 seeds downstream."""
    v2 = df[df["split"] == "calibration_curve_v2"].copy()
    v2["k"] = v2["notes"].str.extract(r";k=(\d)").astype(int)
    v2["draw"] = v2["notes"].str.extract(r"draw=(\d)").fillna(0).astype(int)
    v2["train_key"] = v2["notes"].str.extract(r"train_runs=\[([^\]]*)\]")[0]
    deep = v2[(v2["recipe"] == "dec") & (v2["filter"] == "fir")]
    classical = v2[v2["mode"] == "ts_lr"]
    return pd.concat([deep, classical], ignore_index=True)


# ---------------------------------------------------------------- figure 1: LORO
#
# Leave-one-run-out = the k=7 arm of the leak-free v2 sweep: checkpoint selection
# uses an inner validation split carved from the training runs, the held-out run
# is evaluated exactly once, and deep fits average over 3 seeds. The original
# December leave_one_run_out rows are NOT used: they were inflated by selection
# on the test run and by a load_model cache bug that leaked test-run data into
# the "pretrained" starting weights (see README).

def fig_loro(df):
    loro = v2_rows(df)
    loro = loro[loro["k"] == 7]
    pivot = loro.pivot_table(index="test_run", columns="mode", values="val_accuracy", aggfunc="mean").sort_index()
    summary = loro.groupby("mode")["val_accuracy"].agg(["mean", "std"])

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10.5, 4.0), gridspec_kw={"width_ratios": [2.4, 1.0], "wspace": 0.42}
    )

    x = np.arange(len(pivot.index))
    for mode in ["ts_lr", "transfer", "scratch"]:
        ax1.plot(x, pivot[mode], "-o", color=SERIES[mode], linewidth=2,
                 markersize=7, markeredgecolor="white", markeredgewidth=1.5,
                 label=LABELS[mode], zorder=3)
    ax1.set_xticks(x, pivot.index)
    ax1.set_ylim(0.40, 0.85)
    ax1.set_xlim(-0.4, len(x) - 0.6)
    style_axis(ax1)
    chance_line(ax1, x=len(x) - 0.6)
    ax1.set_xlabel("Held-out run")
    ax1.set_ylabel("Accuracy on held-out run")
    ax1.set_title("Accuracy by held-out run", loc="left")
    ax1.legend(loc="lower left", fontsize=9, handlelength=1.4)

    order = ["ts_lr", "transfer", "scratch"]
    ypos = np.arange(len(order))[::-1]
    for yp, mode in zip(ypos, order):
        m, s = summary.loc[mode, "mean"], summary.loc[mode, "std"]
        ax2.errorbar(m, yp, xerr=s, color=SERIES[mode], fmt="o", markersize=8,
                     markeredgecolor="white", markeredgewidth=1.5, capsize=3,
                     elinewidth=2, zorder=3)
        ax2.annotate(f"{m:.0%}", (m, yp), xytext=(0, 9), textcoords="offset points",
                     ha="center", fontsize=9.5, color=INK, fontweight="bold")
    ax2.axvline(0.5, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    ax2.annotate("chance", (0.5, ypos.max() + 0.55), xytext=(2, 0), textcoords="offset points",
                 color=MUTED, fontsize=8.5, ha="left", va="bottom")
    ax2.set_yticks(ypos, [LABELS[m].split(" (")[0] for m in order])
    ax2.set_ylim(-0.6, len(order) - 0.2)
    ax2.set_xlim(0.40, 0.85)
    ax2.tick_params(axis="y", labelcolor=INK_2)
    style_axis(ax2, grid_axis="x")
    ax2.set_xlabel("Mean accuracy ± s.d.")
    ax2.set_title("Mean across 8 runs", loc="left")

    fig.suptitle("Leave-one-run-out generalization, leak-free protocol (chance = 50%)",
                 x=0.005, y=1.03, ha="left", fontsize=13, fontweight="bold", color=INK)
    save(fig, "loro_generalization.png")
    return summary


# ---------------------------------------------------------------- figure 2: calibration curve
#
# Cross-session protocol (leak-free v2): for each held-out test run, train on k
# runs randomly drawn from the other 7 (2 draws per test run, draws shared
# across methods/seeds so comparisons are paired) and evaluate on the held-out
# run once, with checkpoint selection on an inner validation split. k=7 is
# exactly LORO, so those rows anchor the right edge of the curve.

def fig_calibration(df):
    both = v2_rows(df)
    if both.empty:
        print("skipping calibration curve: no split=calibration_curve_v2 rows in metrics CSV")
        return None

    ks = [1, 2, 4, 7]
    xpos = {k: i for i, k in enumerate(ks)}
    agg = both.groupby(["mode", "k"])["val_accuracy"].agg(["mean", "std", "count"]).reset_index()

    # Paired transfer - scratch gain: identical (test_run, train set, seed) fits
    deep = both[both["mode"].isin(["transfer", "scratch"])].copy()
    deep["seed"] = deep["notes"].apply(lambda s: parse_note(s, "seed"))
    pivot = deep.pivot_table(index=["k", "test_run", "train_key", "seed"], columns="mode",
                             values="val_accuracy", aggfunc="mean").reset_index()
    pivot["gain"] = pivot["transfer"] - pivot["scratch"]
    gain = pivot.groupby("k")["gain"].agg(["mean", "std", "count"]).reset_index()

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.25}
    )

    for mode in ["ts_lr", "transfer", "scratch"]:
        sub = agg[agg["mode"] == mode].sort_values("k")
        x = [xpos[k] for k in sub["k"]]
        ax1.errorbar(x, sub["mean"], yerr=sub["std"], color=SERIES[mode], fmt="-o",
                     linewidth=2, markersize=7, markeredgecolor="white",
                     markeredgewidth=1.5, capsize=3, elinewidth=1.6,
                     label=LABELS[mode], zorder=3)
    ax1.set_xticks(range(len(ks)), ["1 run", "2 runs", "4 runs", "7 runs\n(= LORO)"])
    ax1.set_ylim(0.40, 0.85)
    ax1.set_xlim(-0.35, len(ks) - 0.65)
    style_axis(ax1)
    chance_line(ax1, x=len(ks) - 0.65, ha="right")
    ax1.set_xlabel("Calibration runs used for training")
    ax1.set_ylabel("Accuracy on held-out run (mean ± s.d.)")
    ax1.set_title("Accuracy on an unseen session", loc="left")
    ax1.legend(loc="lower right", fontsize=9)

    from matplotlib.ticker import FuncFormatter

    x = [xpos[k] for k in gain["k"]]
    ax2.axhline(0, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    ax2.annotate("no gain", (len(ks) - 0.7, 0), xytext=(0, 4), textcoords="offset points",
                 color=MUTED, fontsize=8.5, ha="right")
    ax2.errorbar(x, gain["mean"], yerr=gain["std"], color=SERIES["transfer"], fmt="-o",
                 linewidth=2, markersize=7, markeredgecolor="white",
                 markeredgewidth=1.5, capsize=3, elinewidth=1.6, zorder=3)
    for xi, m in zip(x, gain["mean"]):
        pts = f"{m * 100:+.0f}"
        unit = "pt" if abs(round(m * 100)) == 1 else "pts"
        ax2.annotate(f"{pts} {unit}", (xi, m), xytext=(0, 11),
                     textcoords="offset points", ha="center", fontsize=9.5,
                     color=INK, fontweight="bold")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:+.0f}"))
    ax2.set_xticks(range(len(ks)), ["1", "2", "4", "7"])
    ax2.set_xlim(-0.35, len(ks) - 0.65)
    style_axis(ax2)
    ax2.set_xlabel("Calibration runs")
    ax2.set_ylabel("Accuracy gain (percentage points)")
    ax2.set_title("Pretraining gain (paired, transfer − scratch)", loc="left")

    fig.suptitle("Cross-session accuracy vs. amount of calibration data",
                 x=0.005, y=1.03, ha="left", fontsize=13, fontweight="bold", color=INK)
    save(fig, "calibration_curve.png")
    return agg, gain


# ---------------------------------------------------------------- EEG loading helpers

def load_epochs(l_freq=1.0, h_freq=45.0, tmin=0.0, tmax=2.0, montage=True):
    import mne

    epochs_list = []
    for path in sorted(DATA_DIR.glob("sub-01_run-0*_online_raw.fif")):
        raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
        raw.notch_filter(freqs=60, picks="eeg", method="iir", verbose=False)
        raw.filter(l_freq=l_freq, h_freq=h_freq, picks="eeg", method="iir", verbose=False)
        raw.set_eeg_reference("average", verbose=False)
        events = mne.find_events(raw, stim_channel="STI 014", shortest_event=1, verbose=False)
        ep = mne.Epochs(raw, events, event_id={"Left": 1, "Right": 2}, tmin=tmin, tmax=tmax,
                        baseline=None, proj=False, picks="eeg", preload=True, verbose=False)
        epochs_list.append(ep)
    epochs = mne.concatenate_epochs(epochs_list, verbose=False)
    if montage:
        epochs.rename_channels(dict(zip(epochs.ch_names, ELECTRODES_1020)))
        epochs.set_montage(mne.channels.make_standard_montage("standard_1020"))
    return epochs


# ---------------------------------------------------------------- figure 3: ERD spectra

def fig_erd_spectra():
    epochs = load_epochs()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    bands = [(8, 13, "μ"), (13, 30, "β")]

    for ax, ch in zip(axes, ["C3", "C4"]):
        for cls in ["Left", "Right"]:
            psd = epochs[cls].compute_psd(method="welch", fmin=4, fmax=40, picks=[ch],
                                          n_fft=256, n_per_seg=128, n_overlap=64,
                                          verbose=False)
            data = 10 * np.log10(psd.get_data().mean(axis=0)[0] * 1e12)  # dB re 1 uV^2/Hz
            ax.plot(psd.freqs, data, color=CLASS_COLORS[cls], linewidth=2,
                    label=f"{cls}-hand imagery")
        for lo, hi, name in bands:
            ax.axvspan(lo, hi, color=GRID, alpha=0.45, zorder=0)
            ax.annotate(name, ((lo + hi) / 2, ax.get_ylim()[1]), xytext=(0, -12),
                        textcoords="offset points", ha="center", color=MUTED, fontsize=10)
        style_axis(ax)
        hemi = "left motor cortex" if ch == "C3" else "right motor cortex"
        ax.set_title(f"{ch} ({hemi})", loc="left")
        ax.set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Power (dB)")
    axes[0].legend(loc="upper right", fontsize=9)
    fig.suptitle("Class-conditional spectra over sensorimotor electrodes (all 8 runs)",
                 x=0.005, y=1.03, ha="left", fontsize=13, fontweight="bold", color=INK)
    save(fig, "erd_spectra.png")


# ---------------------------------------------------------------- figure 4: CSP patterns

def fig_csp_patterns():
    import mne
    from mne.decoding import CSP

    epochs = load_epochs(l_freq=8.0, h_freq=30.0)
    X = epochs.get_data()
    y = epochs.events[:, 2]
    csp = CSP(n_components=4, log=True, norm_trace=False)
    csp.fit(X, y)
    fig = csp.plot_patterns(epochs.info, components=range(4), ch_type="eeg",
                            cmap="RdBu_r", show=False, colorbar=False)
    fig.set_size_inches(9.0, 2.8)
    for i, ax in enumerate(fig.axes):
        ax.set_title(f"CSP {i + 1}", fontsize=10, color=INK_2)
    save(fig, "csp_patterns.png")


# ---------------------------------------------------------------- figure 5: confusion matrices

def fig_confusions():
    import torch
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import train_test_split

    try:
        from headset_frontend.online_mi_pipeline import EPOCH_TMAX, EPOCH_TMIN
    except ImportError:  # online_mi_pipeline needs brainflow; the constants don't
        EPOCH_TMIN, EPOCH_TMAX = 0.0, 2.0
    from headset_frontend.online_preprocessing import preprocess_epoch_data

    # The checkpoints pickle a CSP fit with an older MNE; newer MNE's __setstate__
    # rejects the old state dict. The fitted attributes (filters_, mean_, std_) are
    # all transform() needs, so restore them leniently.
    import mne.decoding

    def _lenient_setstate(self, state):
        self.__dict__.update(state)

    mne.decoding.CSP.__setstate__ = _lenient_setstate
    from hybrid_cnn_transformer import HybridModelClassifier
    import mne

    raw_data = []
    for path in sorted(DATA_DIR.glob("sub-01_run-0*_online_raw.fif")):
        raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
        events = mne.find_events(raw, stim_channel="STI 014", shortest_event=1, verbose=False)
        X_run, y_run = preprocess_epoch_data(raw, events, EPOCH_TMIN, EPOCH_TMAX)
        raw_data.append((X_run, y_run - 1))
    common = (min(d[0].shape[2] for d in raw_data) // 32) * 32
    X = np.concatenate([d[0][:, :, :common] for d in raw_data])
    y = np.concatenate([d[1] for d in raw_data])
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    device = torch.device("cpu")
    results = {}
    for mode, path in [("scratch", MODELS_DIR / "offline_scratch" / "offline_scratch.pt"),
                       ("transfer", MODELS_DIR / "offline_transfer" / "offline_transfer.pt")]:
        clf, _ = HybridModelClassifier.load_model(str(path), device=device)
        Xv = torch.tensor(X_val, dtype=torch.float32)
        n_ch = getattr(clf.model, "n_channels", Xv.shape[1])
        Xv = Xv[:, :n_ch, :] if Xv.shape[1] >= n_ch else torch.cat(
            [Xv, torch.zeros(Xv.shape[0], n_ch - Xv.shape[1], Xv.shape[2])], dim=1)
        if not getattr(clf, "feature_modules", None):
            clf.use_feature_modules = False
        res = clf.evaluate(Xv, torch.tensor(y_val, dtype=torch.long),
                           subject_indices=torch.zeros(len(y_val), dtype=torch.long))
        results[mode] = (res["accuracy"], confusion_matrix(res["targets"], res["predictions"]))
        print(f"confusions: {mode} val acc {res['accuracy']:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
    classes = ["Left", "Right"]
    for ax, mode in zip(axes, ["scratch", "transfer"]):
        acc, cm = results[mode]
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                color = "white" if norm[i, j] > 0.6 else INK
                ax.annotate(f"{norm[i, j]:.0%}\n(n={cm[i, j]})", (j, i), ha="center",
                            va="center", color=color, fontsize=10)
        ax.set_xticks([0, 1], classes)
        ax.set_yticks([0, 1], classes)
        ax.tick_params(colors=INK_2, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{LABELS[mode].split(' (')[0]} — {acc:.0%}", loc="left", fontsize=10.5)
        ax.set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    fig.suptitle("Validation confusion matrices (random split, all 8 runs)",
                 x=0.005, y=1.04, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    save(fig, "confusion_matrices.png")


# ---------------------------------------------------------------- main

def main():
    df = load_metrics()
    loro_summary = fig_loro(df)
    calib = fig_calibration(df)
    fig_erd_spectra()
    fig_csp_patterns()
    try:
        fig_confusions()
    except Exception as e:  # torch/checkpoint availability varies by machine
        print(f"skipping confusion matrices: {e}")

    print("\n=== headline numbers ===")
    print("LORO mean ± std:")
    print(loro_summary.round(4))
    if calib is not None:
        agg, gain = calib
        print("\nCross-session calibration curve (mode × k):")
        print(agg.round(4))
        print("\nPaired pretraining gain (transfer − scratch):")
        print(gain.round(4))


if __name__ == "__main__":
    main()
