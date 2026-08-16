#!/usr/bin/env python3
"""Per-run signal-quality diagnostics for the self-recorded MI sessions.

For each run, reports:
  - epochs kept / rejected by the deep path's variance-z>2 criterion
  - broadband amplitude (median trial RMS) and raw (pre-notch) 60 Hz line power
  - mu-band (8-13 Hz) ERD lateralization at C3/C4: how much contralateral
    desynchronization distinguishes the classes (positive = classic pattern;
    ~0 = no class-separable motor signal, so no decoder can do well)
  - within-run 5-fold TS+LR CV accuracy at two analysis windows

This is a diagnostic (seconds of CPU), not a training experiment. Results are
printed and written to runs/self/results/run_diagnostics.csv.
"""
import sys
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "apps" / "headset_frontend" / "src"))
sys.path.insert(0, str(REPO / "pipeline" / "src"))
sys.path.insert(0, str(REPO / "pipeline" / "src" / "pipeline" / "public"))
os.environ.setdefault("PYGLET_HEADLESS", "True")

import numpy as np
import pandas as pd
import mne

from headset_frontend.online_preprocessing import preprocess_epoch_data
from pipeline.self.mi_riemann_probe import (
    cv_scores_cov_models,
    load_epochs_for_subject,
    set_true_montage,
)

DATA_DIR = REPO / "data" / "self"
OUT_CSV = REPO / "runs" / "self" / "results" / "run_diagnostics.csv"
RUNS = [f"{i:02d}" for i in range(1, 9)]
MU = (8.0, 13.0)


def band_logpow(ep, ch, band, t0, t1):
    """Mean log mu-band power per trial at one channel over [t0, t1]."""
    e = ep.copy().pick([ch]).filter(band[0], band[1], fir_design="firwin", verbose=False)
    data = e.crop(tmin=t0, tmax=t1).get_data()
    return np.log(np.mean(data ** 2, axis=-1) + 1e-24).ravel()


def main():
    rows = []
    for run in RUNS:
        # --- deep-path epoch rejection + amplitude/line-noise stats
        raw = mne.io.read_raw_fif(str(DATA_DIR / f"sub-01_run-{run}_online_raw.fif"),
                                  preload=True, verbose=False)
        events = mne.find_events(raw, stim_channel="STI 014", shortest_event=1, verbose=False)
        X, y = preprocess_epoch_data(raw.copy(), events, 0.0, 2.0)
        n_events, n_kept = len(events), len(y)
        rms = float(np.median(np.sqrt(np.mean(X ** 2, axis=(1, 2)))))

        psd = raw.compute_psd(picks="eeg", fmin=2, fmax=62, verbose=False)
        p = psd.get_data().mean(axis=0)
        f = psd.freqs
        line = float(p[(f > 58) & (f < 62)].mean() / p[(f > 40) & (f < 55)].mean())

        # --- mu ERD lateralization (baseline = 1 s pre-cue rest)
        ep = load_epochs_for_subject("01", DATA_DIR, epoch_tmin=-1.0, epoch_tmax=2.5,
                                     runs=[run], reject_by_annotation=False)
        ep = set_true_montage(ep)
        y_ep = ep.events[:, -1]  # 1=Left, 2=Right
        erd = {}
        for ch in ("C3", "C4"):
            act = band_logpow(ep, ch, MU, 0.5, 2.5)
            base = band_logpow(ep, ch, MU, -1.0, 0.0)
            erd[ch] = act - base
        # positive = contralateral desync stronger than ipsilateral, both classes
        lat = float((erd["C4"][y_ep == 2].mean() - erd["C4"][y_ep == 1].mean())
                    + (erd["C3"][y_ep == 1].mean() - erd["C3"][y_ep == 2].mean()))

        # --- within-run TS+LR CV accuracy (5-fold)
        accs = {}
        for label, (t0, t1) in (("cv_acc_2s", (0.5, 2.0)), ("cv_acc_2p5s", (0.5, 2.5))):
            e = ep.copy().crop(tmin=t0, tmax=t1)
            _, ts = cv_scores_cov_models(e.get_data(), (y_ep == 2).astype(int))
            accs[label] = float(ts.mean())

        rows.append({
            "run": run,
            "events": n_events,
            "kept": n_kept,
            "rejected": n_events - n_kept,
            "median_rms": rms,
            "raw_line_60hz_x": round(line, 2),
            "mu_erd_lateralization": round(lat, 3),
            **{k: round(v, 3) for k, v in accs.items()},
        })
        print(f"run {run}: kept {n_kept}/{n_events}  line60x{line:.1f}  "
              f"mu-lat {lat:+.3f}  CV(0.5-2.0s) {accs['cv_acc_2s']:.3f}  "
              f"CV(0.5-2.5s) {accs['cv_acc_2p5s']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV.relative_to(REPO)}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
