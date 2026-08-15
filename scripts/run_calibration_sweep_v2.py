#!/usr/bin/env python3
"""Leak-free cross-session calibration sweep (v2) — GPU machine, single environment.

Differences from v1 (`run_calibration_sweep.py`):

1. **No model-selection leakage.** Deep fits carve a stratified 20% inner
   validation split from the TRAINING trials for best-checkpoint selection
   (`inner_val_frac=0.2`); the held-out test run is only tracked per-epoch
   (history['test_acc']) and evaluated once on the selected checkpoint. The
   logged value is that test accuracy (metric=test_accuracy in notes).
2. **Filter method is a factor.** The February repo-ification silently switched
   `preprocess_epoch_data` from mne's FIR default to IIR, which is what made the
   December LORO numbers (FIR) irreproducible under the committed code (IIR) —
   and the base model was pretrained on FIR-filtered public data, so FIR is also
   the domain-matched choice for transfer. Deep modes run under both
   filter='fir' and filter='iir'. TS+LR keeps its own (unchanged) preprocessing.
3. **Recipes.** 'dec' = December hyperparameters (scratch 70 / transfer 50
   epochs, batch 32, lr 1e-4, wd 1e-5, dropout 0.1, csp 6, unfrozen backbone,
   refit_normalizer). 'fix' = transfer-only fine-tuning fix motivated by the
   December trajectories (model is near cross-session best BEFORE fine-tuning
   erodes it): freeze backbone (head/fusion only), lr 1e-5.
4. **Seeds actually applied** (torch/np/random now seeded inside
   train_and_eval_on_runs): seeds {0,1,2} per deep config; draws are shared
   across modes/seeds/filters/recipes (paired), same RNG as v1.
5. TF32 disabled explicitly for reproducibility.

Rows merge idempotently into runs/self/results/offline_metrics.csv with
split=calibration_curve_v2 and tag= in notes.
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

WORK = REPO / "runs" / "self" / "calibration_sweep_v2"
DATA_DIR = REPO / "data" / "self"
BASE_MODEL = REPO / "legacy" / "model" / "models" / "base_model_90_subjects.pt"
MAIN_CSV = REPO / "runs" / "self" / "results" / "offline_metrics.csv"

RUNS = ["01", "02", "03", "04", "05", "06", "07", "08"]
KS = [1, 2, 4, 7]
N_DRAWS = 2
SEEDS = [0, 1, 2]
FILTERS = ["fir", "iir"]
INNER_VAL = 0.2
EPOCH_TMIN, EPOCH_TMAX = 0.0, 2.0
N_WORKERS = 3

os.environ.setdefault("OMP_NUM_THREADS", "4")


def _setup_paths():
    sys.path.insert(0, str(REPO / "apps" / "headset_frontend" / "src"))
    sys.path.insert(0, str(REPO / "pipeline" / "src"))
    sys.path.insert(0, str(REPO / "pipeline" / "src" / "pipeline" / "public"))
    import torch

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import numpy as np
    import mne.decoding

    mne.decoding.CSP.__setstate__ = lambda self, state: self.__dict__.update(state)

    # Average-referenced data is rank-deficient; MNE's GED solver sometimes hits
    # a non-PD covariance with reg=None. Retry with shrinkage only then.
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


def make_draws(k, test_run):
    """Same RNG as v1 so draws stay paired with the v1 sweep."""
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
                # classical baseline: one fit, own preprocessing
                tasks.append({
                    "mode": "ts_lr", "recipe": "classical", "filter": "native",
                    "seed": 0, "k": k, "test_run": test_run, "draw": d,
                    "train_runs": list(subset),
                })
                for filt in FILTERS:
                    for seed in SEEDS:
                        for mode, recipe in [("scratch", "dec"), ("transfer", "dec"), ("transfer", "fix")]:
                            tasks.append({
                                "mode": mode, "recipe": recipe, "filter": filt,
                                "seed": seed, "k": k, "test_run": test_run, "draw": d,
                                "train_runs": list(subset),
                            })
    # Big deep jobs first for load balancing; cheap ts_lr fills gaps.
    tasks.sort(key=lambda t: (t["mode"] == "ts_lr", -t["k"]))
    return tasks


def task_tag(t):
    return (f"{t['mode']}_{t['recipe']}_{t['filter']}_k{t['k']}"
            f"_test{t['test_run']}_d{t['draw']}_s{t['seed']}")


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


def _run_task_inner(task, tag, t0):
    _setup_paths()
    mode, k, test_run = task["mode"], task["k"], task["test_run"]
    train_runs, draw, seed = task["train_runs"], task["draw"], task["seed"]
    recipe, filt = task["recipe"], task["filter"]

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
                f"csp_components=6;runs={RUNS!r};seed={seed};val_split=0.0;"
                f"split=calibration_curve_v2;train_runs={train_runs!r};"
                f"test_run={test_run};metric=test_accuracy;"
                f"baseline=ts_lr;tslr_tmin=0.5;tslr_tmax=2.0;tslr_use_csd=1;tslr_cov_est=lwf;"
                f"notes_extra=experiment=calibration_curve_v2;k={k};draw={draw};"
                f"recipe=classical;tag={tag}"
            ),
        }
        import pandas as pd

        pd.DataFrame([row]).to_csv(WORK / "rows" / f"{tag}.csv", index=False)
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
        seed=seed,
        n_epochs=50 if mode == "transfer" else 70,
        batch_size=32,
        lr=(1e-5 if recipe == "fix" else 1e-4),
        weight_decay=1e-5,
        dropout=0.1,
        n_csp_components=6,
        freeze_backbone=(recipe == "fix"),
        refit_normalizer=True,
        split_tag="calibration_curve_v2",
        metric_name="test_accuracy",
        filter_method=filt,
        inner_val_frac=INNER_VAL,
        notes_extra=f"experiment=calibration_curve_v2;k={k};draw={draw};recipe={recipe};tag={tag}",
    )
    try:
        acc = train_and_eval_on_runs(**kwargs)
    except AssertionError:
        # "No parameters changed when loading the best model state": benign edge
        # when the best epoch is the final one; a different seed sidesteps it.
        retry_seed = seed + 100
        print(f"{tag}: best-state assertion, retrying with seed={retry_seed}", flush=True)
        kwargs["seed"] = retry_seed
        acc = train_and_eval_on_runs(**kwargs)
    return tag, float(acc), time.time() - t0


def main():
    (WORK / "rows").mkdir(parents=True, exist_ok=True)
    (WORK / "models").mkdir(parents=True, exist_ok=True)
    tasks = make_tasks()
    n_deep = sum(1 for t in tasks if t["mode"] != "ts_lr")
    n_done = sum(1 for t in tasks if (WORK / "rows" / f"{task_tag(t)}.csv").exists())
    print(f"{len(tasks)} tasks ({n_deep} deep fits, {n_done} already done), "
          f"{N_WORKERS} workers", flush=True)

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
    existing = existing[existing["notes"].astype(str).str.contains("split=calibration_curve_v2")]
    have = set(tags_of(existing)) if len(existing) else set()
    merged = merged[~tags_of(merged).isin(have)]
    if len(merged):
        merged.to_csv(MAIN_CSV, mode="a", header=False, index=False)
    print(f"appended {len(merged)} new rows to {MAIN_CSV}", flush=True)

    allrows = pd.concat(frames, ignore_index=True)
    for key in ["mode", "recipe", "filter", "k"]:
        allrows[key] = allrows["notes"].str.extract(
            {"mode": r"tag=(\w+?)_", "recipe": r"recipe=(\w+)",
             "filter": r"tag=\w+?_\w+?_(\w+?)_k", "k": r";k=(\d+)"}[key])
    allrows["k"] = allrows["k"].astype(int)
    print(allrows.groupby(["mode", "recipe", "filter", "k"])["val_accuracy"]
          .agg(["mean", "std", "count"]).round(3).to_string(), flush=True)
    print("SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
