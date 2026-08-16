#!/usr/bin/env python3
"""Improvement sweep: session alignment, full imagery window, filter bank,
channel hygiene, and CSD — evaluated with the same paired, leakage-controlled
cross-session protocol as run_calibration_sweep_v2.py.

Arms (classical TS+LR unless noted; every arm uses the identical train/test
draws as the v2 sweep, so all comparisons are paired):

  ts_base        current baseline replicated (0.5-2.0 s, broadband 8-30, all
                 channels, no alignment) — must match the v2 ts_lr rows exactly
  ts_align_e     + session-wise Euclidean covariance alignment
  ts_align_r     + session-wise Riemannian (geometric-mean) alignment
  ts_win         + full imagery window (0.5-2.5 s; the online recordings hold
                 2.5 s of imagery per trial, previous analyses used 2.0 s)
  ts_fb          + filter bank (8-12, 12-16, 16-22, 22-30 Hz tangent vectors)
  ts_chan        + drop frontal channels (Fp1, Fp2, F7, F8; CSP patterns show
                 ocular contamination, which should not survive session drift)
  ts_csd         + real surface Laplacian (montage attached; the old use_csd
                 flag was a silent no-op because the FIFs carry no montage)
  ts_stack       riemann align + full window + filter bank
  ts_stack_chan  ts_stack + frontal channels dropped
  transfer_ea    deep transfer (Dec recipe, FIR, inner-val) + per-session
                 Euclidean alignment of the raw trials, 3 seeds

Rows merge idempotently into runs/self/results/offline_metrics.csv with
split=improvement_v1 and tag= in notes. Resume-safe: one CSV per finished fit
under runs/self/improvement_sweep/rows/.

Run from the repo root:  .venv/bin/python scripts/run_improvement_sweep.py
"""
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

WORK = REPO / "runs" / "self" / "improvement_sweep"
DATA_DIR = REPO / "data" / "self"
BASE_MODEL = REPO / "legacy" / "model" / "models" / "base_model_90_subjects.pt"
MAIN_CSV = REPO / "runs" / "self" / "results" / "offline_metrics.csv"

RUNS = ["01", "02", "03", "04", "05", "06", "07", "08"]
KS = [1, 2, 4, 7]
N_DRAWS = 2
DEEP_SEEDS = [0, 1, 2]
INNER_VAL = 0.2
SPLIT_TAG = "improvement_v1"
N_WORKERS = 12

FB_BANDS = ((8, 12), (12, 16), (16, 22), (22, 30))
FRONTAL = ("Fp1", "Fp2", "F7", "F8")

# arm name -> ts_lr_runwise_accuracy_v2 kwargs (classical arms only)
CLASSICAL_ARMS = {
    "ts_base":       dict(tmin=0.5, tmax=2.0),
    "ts_align_e":    dict(tmin=0.5, tmax=2.0, align="euclid"),
    "ts_align_r":    dict(tmin=0.5, tmax=2.0, align="riemann"),
    "ts_win":        dict(tmin=0.5, tmax=2.5),
    "ts_fb":         dict(tmin=0.5, tmax=2.0, bands=FB_BANDS),
    "ts_chan":       dict(tmin=0.5, tmax=2.0, drop_channels=FRONTAL),
    "ts_csd":        dict(tmin=0.5, tmax=2.0, use_csd=True),
    "ts_stack":      dict(tmin=0.5, tmax=2.5, align="riemann", bands=FB_BANDS),
    "ts_stack_chan": dict(tmin=0.5, tmax=2.5, align="riemann", bands=FB_BANDS,
                          drop_channels=FRONTAL),
    # First round showed the 2.5 s window is harmful (the discriminative signal
    # dies after ~2 s) and it dragged down both stacks; retest the combinations
    # at the standard 2.0 s window.
    "ts_stack2":      dict(tmin=0.5, tmax=2.0, align="riemann", bands=FB_BANDS),
    "ts_stack2_chan": dict(tmin=0.5, tmax=2.0, align="riemann", bands=FB_BANDS,
                           drop_channels=FRONTAL),
    "ts_align_chan":  dict(tmin=0.5, tmax=2.0, align="riemann",
                           drop_channels=FRONTAL),
}

os.environ.setdefault("OMP_NUM_THREADS", "2")


def _setup_paths():
    sys.path.insert(0, str(REPO / "apps" / "headset_frontend" / "src"))
    sys.path.insert(0, str(REPO / "pipeline" / "src"))
    sys.path.insert(0, str(REPO / "pipeline" / "src" / "pipeline" / "public"))
    import torch

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import numpy as np
    import mne.decoding

    mne.decoding.CSP.__setstate__ = lambda self, state: self.__dict__.update(state)

    _orig_fit = mne.decoding.CSP.fit

    def _robust_fit(self, X, y):
        try:
            return _orig_fit(self, X, y)
        except np.linalg.LinAlgError:
            print("CSP fit: singular covariance, refitting with reg='ledoit_wolf'", flush=True)
            fresh = mne.decoding.CSP(
                n_components=self.n_components, reg="ledoit_wolf",
                log=self.log, norm_trace=self.norm_trace,
            )
            _orig_fit(fresh, X, y)
            self.__dict__.update(fresh.__dict__)
            return self

    mne.decoding.CSP.fit = _robust_fit

    # feature_utils calls mne filter_data(..., n_jobs=-1); one joblib pool per
    # worker thrashes the box, so force serial filtering inside workers.
    import feature_utils

    _orig_filter_data = feature_utils.filter_data

    def _serial_filter_data(*args, **kwargs):
        kwargs["n_jobs"] = 1
        return _orig_filter_data(*args, **kwargs)

    feature_utils.filter_data = _serial_filter_data


def make_draws(k, test_run):
    """Same RNG as the v1/v2 sweeps so all arms stay paired across sweeps."""
    others = [r for r in RUNS if r != test_run]
    rng = random.Random(1000 * k + int(test_run))
    n_draws = min(N_DRAWS, math.comb(len(others), k))
    draws = set()
    while len(draws) < n_draws:
        draws.add(tuple(sorted(rng.sample(others, k))))
    return sorted(draws)


def make_tasks():
    tasks = []
    for k in KS:
        for test_run in RUNS:
            for d, subset in enumerate(make_draws(k, test_run)):
                for arm in CLASSICAL_ARMS:
                    tasks.append({
                        "arm": arm, "seed": 0, "k": k, "test_run": test_run,
                        "draw": d, "train_runs": list(subset),
                    })
                for seed in DEEP_SEEDS:
                    tasks.append({
                        "arm": "transfer_ea", "seed": seed, "k": k,
                        "test_run": test_run, "draw": d, "train_runs": list(subset),
                    })
    # Deep jobs first for GPU/CPU load balancing; classical fills gaps.
    tasks.sort(key=lambda t: (t["arm"] != "transfer_ea", -t["k"]))
    return tasks


def task_tag(t):
    return f"{t['arm']}_k{t['k']}_test{t['test_run']}_d{t['draw']}_s{t['seed']}"


def run_task(task):
    tag = task_tag(task)
    t0 = time.time()
    row_csv = WORK / "rows" / f"{tag}.csv"
    if row_csv.exists():  # resume support
        import pandas as pd

        return tag, float(pd.read_csv(row_csv)["val_accuracy"].iloc[0]), 0.0
    try:
        return _run_task_inner(task, tag, t0)
    except Exception as e:
        print(f"TASK FAILED {tag}: {type(e).__name__}: {e}", flush=True)
        return tag, float("nan"), time.time() - t0
    finally:
        # Unbounded caches OOM long-lived workers; keep memory flat per task.
        try:
            import feature_utils
            feature_utils._fb_filter_data_cache.clear()
        except Exception:
            pass
        try:
            import hybrid_cnn_transformer as _hct
            _hct._load_model_cache.clear()
        except Exception:
            pass


def _run_task_inner(task, tag, t0):
    _setup_paths()
    arm, k, test_run = task["arm"], task["k"], task["test_run"]
    train_runs, draw, seed = task["train_runs"], task["draw"], task["seed"]

    if arm == "transfer_ea":
        from pipeline.self.train_offline_model import train_and_eval_on_runs

        kwargs = dict(
            data_dir=DATA_DIR,
            subject="01",
            train_runs=train_runs,
            test_runs=[test_run],
            metrics_csv=WORK / "rows" / f"{tag}.csv",
            output_dir=WORK / "models" / tag,
            model_name=tag,
            base_model_path=BASE_MODEL,
            seed=seed,
            n_epochs=50,
            batch_size=32,
            lr=1e-4,
            weight_decay=1e-5,
            dropout=0.1,
            n_csp_components=6,
            freeze_backbone=False,
            refit_normalizer=True,
            split_tag=SPLIT_TAG,
            metric_name="test_accuracy",
            filter_method="fir",
            inner_val_frac=INNER_VAL,
            session_align=True,
            notes_extra=(f"experiment={SPLIT_TAG};k={k};draw={draw};"
                         f"arm={arm};tag={tag}"),
        )
        try:
            acc = train_and_eval_on_runs(**kwargs)
        except AssertionError:
            retry_seed = seed + 100
            print(f"{tag}: best-state assertion, retrying with seed={retry_seed}", flush=True)
            kwargs["seed"] = retry_seed
            acc = train_and_eval_on_runs(**kwargs)
        return tag, float(acc), time.time() - t0

    from pipeline.self.mi_riemann_probe import (
        load_epochs_for_subject,
        ts_lr_runwise_accuracy_v2,
    )

    opts = CLASSICAL_ARMS[arm]
    epoch_tmax = float(opts.get("tmax", 2.0))
    # reject_by_annotation=False: the FIFs carry per-trial buffer-boundary
    # annotations although the stream is continuous; windows past 2.0 s would
    # otherwise lose most epochs. Used for every arm so trial sets stay paired.
    ep_tr = load_epochs_for_subject("01", DATA_DIR, epoch_tmin=0.0,
                                    epoch_tmax=epoch_tmax, runs=train_runs,
                                    reject_by_annotation=False)
    ep_te = load_epochs_for_subject("01", DATA_DIR, epoch_tmin=0.0,
                                    epoch_tmax=epoch_tmax, runs=[test_run],
                                    reject_by_annotation=False)
    acc = float(ts_lr_runwise_accuracy_v2(ep_tr, ep_te, cov_est="lwf", **opts))

    opts_note = ";".join(
        f"tslr_{key}={val}" for key, val in sorted(opts.items()) if val is not None
    ).replace(" ", "")
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "mode": "ts_lr",
        "base_model": "",
        "subjects": 1,
        "trials": int(len(ep_tr) + len(ep_te)),
        "val_accuracy": acc,
        "notes": (
            f"csp_components=6;runs={RUNS!r};seed={seed};val_split=0.0;"
            f"split={SPLIT_TAG};train_runs={train_runs!r};"
            f"test_run={test_run};metric=test_accuracy;baseline=ts_lr;"
            f"{opts_note};cov_est=lwf;"
            f"notes_extra=experiment={SPLIT_TAG};k={k};draw={draw};"
            f"arm={arm};tag={tag}"
        ),
    }
    import pandas as pd

    pd.DataFrame([row]).to_csv(WORK / "rows" / f"{tag}.csv", index=False)
    return tag, acc, time.time() - t0


def main():
    (WORK / "rows").mkdir(parents=True, exist_ok=True)
    (WORK / "models").mkdir(parents=True, exist_ok=True)
    tasks = make_tasks()
    n_deep = sum(1 for t in tasks if t["arm"] == "transfer_ea")
    n_done = sum(1 for t in tasks if (WORK / "rows" / f"{task_tag(t)}.csv").exists())
    print(f"{len(tasks)} tasks ({n_deep} deep, {len(tasks) - n_deep} classical, "
          f"{n_done} already done), {N_WORKERS} workers", flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        for i, (tag, acc, dt) in enumerate(pool.map(run_task, tasks)):
            print(f"[{i + 1}/{len(tasks)}] {tag} acc={acc:.3f} ({dt:.0f}s, "
                  f"elapsed {(time.time() - t0) / 60:.1f}m)", flush=True)

    # Merge worker rows into the main metrics CSV (idempotent, keyed on tag=).
    import pandas as pd
    import re

    frames = [pd.read_csv(p) for p in sorted((WORK / "rows").glob("*.csv"))]
    merged = pd.concat(frames, ignore_index=True)
    cols = list(pd.read_csv(MAIN_CSV, nrows=1).columns)
    merged = merged[cols]

    def tags_of(frame):
        return frame["notes"].map(lambda n: (re.search(r"tag=([^;\"]+)", str(n)) or [None, ""])[1])

    existing = pd.read_csv(MAIN_CSV)
    existing = existing[existing["notes"].astype(str).str.contains(f"split={SPLIT_TAG}")]
    have = set(tags_of(existing)) if len(existing) else set()
    merged = merged[~tags_of(merged).isin(have)]
    if len(merged):
        merged.to_csv(MAIN_CSV, mode="a", header=False, index=False)
    print(f"appended {len(merged)} new rows to {MAIN_CSV}", flush=True)

    allrows = pd.concat(frames, ignore_index=True)
    allrows["arm"] = allrows["notes"].str.extract(r"arm=([^;\"]+)")
    allrows["k"] = allrows["notes"].str.extract(r";k=(\d)").astype(int)
    print(allrows.groupby(["arm", "k"])["val_accuracy"]
          .agg(["mean", "std", "count"]).round(3).to_string(), flush=True)
    print("SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
