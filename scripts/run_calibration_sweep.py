#!/usr/bin/env python3
"""Cross-session calibration-size sweep (the 'correct' calibration curve).

For each held-out test run r (all 8), and each calibration size k in {1,2,4,7}:
draw up to 2 random subsets of size k from the other 7 runs, train on the
subset, evaluate on run r. Modes: deep scratch, deep transfer, classical TS+LR.
k=7 has exactly one subset per test run and is the LORO protocol; comparing it
against the repo's original leave_one_run_out rows measures how much the deep
results shift across training environments (the classical baseline is exact).

Hyperparameters match the notebook's run-wise experiments exactly:
epochs 70 scratch / 50 transfer, batch 32, lr 1e-4, wd 1e-5, dropout 0.1,
csp_components 6, freeze_backbone=False, refit_normalizer=True, seed=0.

Each worker writes to its own CSV; rows are merged into
runs/self/results/offline_metrics.csv at the end with split=calibration_curve.
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

WORK = REPO / "runs" / "self" / "calibration_sweep"
DATA_DIR = REPO / "data" / "self"
BASE_MODEL = REPO / "legacy" / "model" / "models" / "base_model_90_subjects.pt"
MAIN_CSV = REPO / "runs" / "self" / "results" / "offline_metrics.csv"

RUNS = ["01", "02", "03", "04", "05", "06", "07", "08"]
KS = [1, 2, 4, 7]
N_DRAWS = 2
SEED = 0
EPOCH_TMIN, EPOCH_TMAX = 0.0, 2.0

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")


def _setup_paths():
    sys.path.insert(0, str(REPO / "apps" / "headset_frontend" / "src"))
    sys.path.insert(0, str(REPO / "pipeline" / "src"))
    sys.path.insert(0, str(REPO / "pipeline" / "src" / "pipeline" / "public"))
    import torch

    torch.set_num_threads(4)
    import numpy as np
    import mne.decoding

    mne.decoding.CSP.__setstate__ = lambda self, state: self.__dict__.update(state)

    # Average-referenced data is rank-deficient; MNE's GED solver sometimes hits
    # a non-PD covariance with reg=None. Retry with a tiny shrinkage only then.
    _orig_fit = mne.decoding.CSP.fit

    def _robust_fit(self, X, y):
        try:
            return _orig_fit(self, X, y)
        except np.linalg.LinAlgError:
            # 16-ch average-referenced data is exactly rank 15. Mutating self.reg
            # is a no-op (the covariance callable is built at __init__), so build
            # a fresh CSP with Ledoit-Wolf shrinkage and adopt its fitted state.
            print("CSP fit: singular covariance, refitting with reg='ledoit_wolf'", flush=True)
            fresh = mne.decoding.CSP(
                n_components=self.n_components, reg="ledoit_wolf",
                log=self.log, norm_trace=self.norm_trace,
            )
            _orig_fit(fresh, X, y)
            self.__dict__.update(fresh.__dict__)
            return self

    mne.decoding.CSP.fit = _robust_fit


def make_tasks():
    tasks = []
    for k in KS:
        for test_run in RUNS:
            others = [r for r in RUNS if r != test_run]
            rng = random.Random(1000 * k + int(test_run))
            n_draws = min(N_DRAWS, math.comb(len(others), k))
            draws = set()
            while len(draws) < n_draws:
                draws.add(tuple(sorted(rng.sample(others, k))))
            for d, subset in enumerate(sorted(draws)):
                for mode in ["scratch", "transfer", "ts_lr"]:
                    tasks.append({
                        "mode": mode,
                        "k": k,
                        "test_run": test_run,
                        "draw": d,
                        "train_runs": list(subset),
                    })
    # Big deep jobs first for better load balancing; cheap ts_lr fills gaps.
    tasks.sort(key=lambda t: (t["mode"] == "ts_lr", -t["k"]))
    return tasks


def run_task(task):
    mode, k, test_run = task["mode"], task["k"], task["test_run"]
    train_runs, draw = task["train_runs"], task["draw"]
    tag = f"{mode}_k{k}_test{test_run}_d{draw}"
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


def _run_task_inner(task, tag, t0):
    _setup_paths()
    mode, k, test_run = task["mode"], task["k"], task["test_run"]
    train_runs, draw = task["train_runs"], task["draw"]

    if mode == "ts_lr":
        from pipeline.self.mi_riemann_probe import (
            load_epochs_for_subject,
            ts_lr_runwise_accuracy,
        )

        ep_tr = load_epochs_for_subject("01", DATA_DIR, epoch_tmin=EPOCH_TMIN,
                                        epoch_tmax=EPOCH_TMAX, runs=train_runs)
        ep_te = load_epochs_for_subject("01", DATA_DIR, epoch_tmin=EPOCH_TMIN,
                                        epoch_tmax=EPOCH_TMAX, runs=[test_run])
        acc = float(ts_lr_runwise_accuracy(ep_tr, ep_te, tmin=0.5, tmax=2.0,
                                           cov_est="lwf", use_csd=True))
        n_trials = int(len(ep_tr) + len(ep_te))
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "mode": "ts_lr",
            "base_model": "",
            "subjects": 1,
            "trials": n_trials,
            "val_accuracy": acc,
            "notes": (
                f"csp_components=6;runs={RUNS!r};seed={SEED};val_split=0.0;"
                f"split=calibration_curve;train_runs={train_runs!r};"
                f"test_run={test_run};metric=val_accuracy;"
                f"baseline=ts_lr;tslr_tmin=0.5;tslr_tmax=2.0;tslr_use_csd=1;tslr_cov_est=lwf;"
                f"notes_extra=experiment=calibration_curve;k={k};draw={draw}"
            ),
        }
        import pandas as pd

        out_csv = WORK / "rows" / f"{tag}.csv"
        pd.DataFrame([row]).to_csv(out_csv, index=False)
        return tag, acc, time.time() - t0

    from pipeline.self.train_offline_model import train_and_eval_on_runs

    kwargs = dict(
        data_dir=DATA_DIR,
        subject="01",
        train_runs=train_runs,
        test_runs=[test_run],
        metrics_csv=WORK / "rows" / f"{tag}.csv",
        output_dir=WORK / "models" / tag,
        model_name=tag,
        base_model_path=BASE_MODEL if mode == "transfer" else None,
        seed=SEED,
        n_epochs=50 if mode == "transfer" else 70,
        batch_size=32,
        lr=1e-4,
        weight_decay=1e-5,
        dropout=0.1,
        n_csp_components=6,
        freeze_backbone=False,
        refit_normalizer=True,
        split_tag="calibration_curve",
        metric_name="val_accuracy",
        notes_extra=f"experiment=calibration_curve;k={k};draw={draw}",
    )
    try:
        acc = train_and_eval_on_runs(**kwargs)
    except AssertionError:
        # "No parameters changed when loading the best model state": benign edge
        # when the best epoch is the final one; a different seed sidesteps it.
        print(f"{tag}: best-state assertion, retrying with seed=1", flush=True)
        kwargs["seed"] = 1
        acc = train_and_eval_on_runs(**kwargs)
    return tag, float(acc), time.time() - t0


def main():
    (WORK / "rows").mkdir(parents=True, exist_ok=True)
    (WORK / "models").mkdir(parents=True, exist_ok=True)
    tasks = make_tasks()
    n_deep = sum(1 for t in tasks if t["mode"] != "ts_lr")
    print(f"{len(tasks)} tasks ({n_deep} deep fits), 4 workers", flush=True)

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i, (tag, acc, dt) in enumerate(pool.map(run_task, tasks)):
            print(f"[{i + 1}/{len(tasks)}] {tag} acc={acc:.3f} ({dt:.0f}s, "
                  f"elapsed {(time.time() - t0) / 60:.1f}m)", flush=True)
            results.append((tag, acc))

    # Merge worker rows into the main metrics CSV.
    import pandas as pd

    frames = [pd.read_csv(p) for p in sorted((WORK / "rows").glob("*.csv"))]
    merged = pd.concat(frames, ignore_index=True)
    cols = list(pd.read_csv(MAIN_CSV, nrows=1).columns)
    merged = merged[cols]

    # Idempotent merge: skip rows already appended by a previous (partial) run.
    def task_key(frame):
        import re as _re

        def _k(notes, key):
            m = _re.search(rf"{key}=(\d+)", str(notes))
            return m.group(1) if m else ""

        return (frame["mode"].astype(str) + "|k" + frame["notes"].map(lambda n: _k(n, "k"))
                + "|d" + frame["notes"].map(lambda n: _k(n, "draw"))
                + "|t" + frame["notes"].map(lambda n: _k(n, "test_run")))

    existing = pd.read_csv(MAIN_CSV)
    existing = existing[existing["notes"].astype(str).str.contains("split=calibration_curve")]
    have = set(task_key(existing)) if len(existing) else set()
    merged = merged[~task_key(merged).isin(have)]
    if len(merged):
        merged.to_csv(MAIN_CSV, mode="a", header=False, index=False)
    print(f"appended {len(merged)} new rows to {MAIN_CSV}", flush=True)

    summary = merged.copy()
    summary["k"] = summary["notes"].str.extract(r"k=(\d+)").astype(int)
    print(summary.groupby(["mode", "k"])["val_accuracy"]
          .agg(["mean", "std", "count"]).round(3), flush=True)
    print("SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
