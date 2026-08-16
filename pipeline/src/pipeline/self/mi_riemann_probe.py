#!/usr/bin/env python3
import argparse, numpy as np, mne, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle as skshuffle
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM
from sklearn.model_selection import GroupKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd

def load_epochs_from_input(path, epoch_tmin, epoch_tmax):
    """
    Load MNE Epochs from a path that may be either an epochs .fif file
    or a raw .fif file saved from the online pipeline. If raw, find events
    on 'STI 014' and construct epochs with Left/Right mapping.
    """
    try:
        # Try direct epochs read first
        return mne.read_epochs(path, preload=True, verbose='ERROR')
    except Exception:
        # Fallback: treat as raw and build epochs like train_offline_model
        raw = mne.io.read_raw_fif(str(path), preload=True, verbose=False)
        raw = preprocess_raw(raw)
        events = mne.find_events(raw, stim_channel='STI 014', shortest_event=1)
        epochs = mne.Epochs(
            raw,
            events,
            event_id={"Left": 1, "Right": 2},
            tmin=epoch_tmin,
            tmax=epoch_tmax,
            baseline=None,
            proj=False,
            picks="eeg",
            preload=True,
            verbose=False,
            on_missing='ignore'
        )
        return epochs

def load_epochs_for_subject(subject, data_dir, epoch_tmin, epoch_tmax, runs=None,
                            reject_by_annotation=True):
    """
    Load and concatenate epochs across all runs for a subject from raw FIFs
    saved by the online pipeline (pattern: sub-<ID>_run-XX_online_raw.fif).

    The online pipeline saved each session as concatenated buffer chunks, so the
    FIFs carry per-trial 'BAD boundary' annotations even though the underlying
    stream is continuous (inter-event sample spacing matches the real-time trial
    period). With the default reject_by_annotation=True, windows longer than
    ~2 s post-cue lose most epochs to those annotations; pass False to epoch
    across them (needed for any analysis window beyond 2.0 s).
    """
    data_dir = Path(data_dir)
    if runs is None:
        files = sorted(data_dir.glob(f"sub-{subject}_run-*_online_raw.fif"))
    else:
        files = [data_dir / f"sub-{subject}_run-{int(r):02d}_online_raw.fif" for r in runs]
        files = [f for f in files if f.exists()]
    if not files:
        raise FileNotFoundError(f"No runs found for subject {subject} in {data_dir}")
    epochs_list = []
    for f in files:
        raw = mne.io.read_raw_fif(str(f), preload=True, verbose=False)
        raw = preprocess_raw(raw)
        events = mne.find_events(raw, stim_channel='STI 014', shortest_event=1)
        ep = mne.Epochs(
            raw,
            events,
            event_id={"Left": 1, "Right": 2},
            tmin=epoch_tmin,
            tmax=epoch_tmax,
            baseline=None,
            proj=False,
            picks="eeg",
            preload=True,
            verbose=False,
            on_missing='ignore',
            reject_by_annotation=reject_by_annotation,
        )
        run_id = int(str(f).split('_run-')[1][:2])
        ep.metadata = pd.DataFrame({"run": np.full(len(ep), run_id)})
        if len(ep) > 0:
            epochs_list.append(ep)
    if not epochs_list:
        raise RuntimeError(f"No epochs created for subject {subject} (check events/time window)")
    return mne.concatenate_epochs(epochs_list)

def pick_left_right(epochs):
    ev = epochs.event_id
    keys = list(ev.keys())
    lk = [k for k in keys if 'left' in k.lower()]
    rk = [k for k in keys if 'right' in k.lower()]
    if lk and rk:
        return [lk[0], rk[0]]
    # fallback: take first two classes
    assert len(keys) >= 2, "Need at least two classes"
    return keys[:2]

def preprocess_raw(raw, notch=60, l_freq=8, h_freq=30):
    """
    Preprocess Raw before epoching:
    - Average reference (EEG)
    - Notch at powerline and first harmonic (EEG picks only)
    - Band-pass to motor band (EEG picks only)
    """
    raw = raw.copy().load_data()
    raw.set_eeg_reference('average')
    try:
        raw.notch_filter(freqs=[notch, 2 * notch], picks='eeg', method='iir')
    except Exception:
        raw.notch_filter(freqs=[notch], picks='eeg', method='iir')
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', picks='eeg')
    return raw

def preprocess(epochs, notch=60, l_freq=8, h_freq=30, use_csd=True):
    # Minimal epoch-level processing; filtering is done at Raw level
    ep = epochs.copy().load_data()
    # Optional: surface Laplacian (CSD) if requested and montage present
    if use_csd and ep.get_montage() is not None:
        try:
            ep = mne.preprocessing.compute_current_source_density(ep)
        except Exception:
            pass
    return ep

def crop_and_get_Xy(epochs, classes, tmin, tmax):
    ep = epochs[classes].copy().crop(tmin=tmin, tmax=tmax)
    X = ep.get_data()                 # (n_epochs, n_ch, n_times)
    y = ep.events[:, -1]              # numeric labels per event_id
    # map to 0/1 in class order
    id_map = {epochs.event_id[c]: i for i, c in enumerate(classes)}
    y = np.vectorize(id_map.get)(y)
    return ep, X, y

def cv_scores_cov_models(X, y, cov_est='lwf', random_state=13):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    mdm = MDM(metric='riemann')
    ts = TangentSpace(metric='riemann')
    lr = LogisticRegression(max_iter=2000, n_jobs=None, solver='lbfgs')
    cov = Covariances(estimator=cov_est)

    acc_mdm, acc_ts = [], []
    for tr, te in skf.split(X, y):
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        Ctr, Cte = cov.fit_transform(Xtr), cov.transform(Xte)
        # MDM
        mdm.fit(Ctr, ytr)
        acc_mdm.append(accuracy_score(yte, mdm.predict(Cte)))
        # Tangent-space + LR
        Ztr, Zte = ts.fit_transform(Ctr), ts.transform(Cte)
        lr.fit(Ztr, ytr)
        acc_ts.append(accuracy_score(yte, lr.predict(Zte)))
    return np.array(acc_mdm), np.array(acc_ts)


def fit_ts_lr(X_train, y_train, cov_est: str = "lwf"):
    """
    Fit TangentSpace + LogisticRegression on raw epochs arrays.

    Returns a small dict model with keys: cov, ts, lr.
    """
    cov = Covariances(estimator=cov_est)
    Ctr = cov.fit_transform(X_train)
    ts = TangentSpace(metric="riemann")
    Ztr = ts.fit_transform(Ctr)
    lr = LogisticRegression(max_iter=2000, solver="lbfgs")
    lr.fit(Ztr, y_train)
    return {"cov": cov, "ts": ts, "lr": lr}


def score_ts_lr(model, X_test, y_test):
    """Score a fitted TS+LR model on a held-out set; returns (acc, y_pred)."""
    Cte = model["cov"].transform(X_test)
    Zte = model["ts"].transform(Cte)
    y_pred = model["lr"].predict(Zte)
    return float(accuracy_score(y_test, y_pred)), y_pred


def ts_lr_runwise_accuracy(
    epochs_train,
    epochs_test,
    *,
    tmin: float = 0.5,
    tmax: float = 2.0,
    cov_est: str = "lwf",
    use_csd: bool = True,
    classes=None,
):
    """
    Train TS+LR on `epochs_train` and evaluate on `epochs_test` (single split).
    Returns accuracy.
    """
    if classes is None:
        classes = pick_left_right(epochs_train)
    ep_tr = preprocess(epochs_train, use_csd=use_csd)
    ep_te = preprocess(epochs_test, use_csd=use_csd)
    _, Xtr, ytr = crop_and_get_Xy(ep_tr, classes, tmin, tmax)
    _, Xte, yte = crop_and_get_Xy(ep_te, classes, tmin, tmax)
    model = fit_ts_lr(Xtr, ytr, cov_est=cov_est)
    acc, _ = score_ts_lr(model, Xte, yte)
    return acc


# ---- v2: session alignment, filter bank, channel picks, real CSD ----------

# OpenBCI Cyton+Daisy default 10-20 layout in board channel order; the committed
# FIFs use generic names C1..C16 (same mapping as legacy/experiment/evaluate_mi_data.py
# and scripts/generate_report_figures.py).
ELECTRODES_1020 = [
    "Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2",
    "F7", "F8", "F3", "F4", "T7", "T8", "P3", "P4",
]
FRONTAL_CHANNELS = ["Fp1", "Fp2", "F7", "F8"]


def set_true_montage(epochs):
    """Rename generic C1..C16 channels to the physical 10-20 layout and attach
    the standard montage (needed for a real CSD and for name-based picks)."""
    ep = epochs.copy()
    eeg_names = [ch for ch in ep.ch_names if ch.startswith("C") and ch[1:].isdigit()]
    if eeg_names:
        mapping = {ch: ELECTRODES_1020[int(ch[1:]) - 1] for ch in eeg_names}
        ep.rename_channels(mapping)
        ep.set_montage(mne.channels.make_standard_montage("standard_1020"),
                       on_missing="ignore")
    return ep


def _session_references(covs, groups, metric):
    """Mean covariance per session (unlabeled). metric: 'euclid' | 'riemann'."""
    from pyriemann.utils.mean import mean_euclid, mean_riemann

    mean_fn = mean_euclid if metric == "euclid" else mean_riemann
    return {g: mean_fn(covs[groups == g]) for g in np.unique(groups)}


def _align_covs(covs, groups, refs):
    """Recenter each trial covariance by its session reference: R^-1/2 C R^-1/2."""
    from pyriemann.utils.base import invsqrtm

    out = np.empty_like(covs)
    for g, ref in refs.items():
        w = invsqrtm(ref)
        idx = np.where(groups == g)[0]
        out[idx] = w @ covs[idx] @ w
    return out


def ts_lr_runwise_accuracy_v2(
    epochs_train,
    epochs_test,
    *,
    tmin: float = 0.5,
    tmax: float = 2.0,
    cov_est: str = "lwf",
    align: str = None,          # None | 'euclid' | 'riemann' (session-wise recentering)
    bands=None,                 # None => broadband; else [(lo, hi), ...] filter bank
    drop_channels=None,         # 10-20 names to exclude (e.g. FRONTAL_CHANNELS)
    use_csd: bool = False,      # real CSD (montage is attached first)
    classes=None,
):
    """
    TS+LR with optional session-wise covariance alignment, filter bank,
    channel exclusion, and a real surface-Laplacian (CSD).

    Alignment recenters each session's trial covariances at that session's own
    (unlabeled) mean, so the test session is aligned without using its labels —
    the standard Euclidean/Riemannian alignment setup for cross-session MI.
    Session membership comes from epochs.metadata['run'].
    """
    if classes is None:
        classes = pick_left_right(epochs_train)

    def prepare(epochs):
        ep = set_true_montage(epochs)
        if drop_channels:
            keep = [ch for ch in ep.ch_names if ch not in set(drop_channels)]
            ep.pick(keep)
        if use_csd:
            ep = mne.preprocessing.compute_current_source_density(ep)
        groups = (ep.metadata["run"].to_numpy() if ep.metadata is not None
                  and "run" in ep.metadata else np.zeros(len(ep), dtype=int))
        # class subsetting must subset groups identically
        sel = np.isin(ep.events[:, -1], [ep.event_id[c] for c in classes])
        ep2, X, y = crop_and_get_Xy(ep, classes, tmin, tmax)
        return ep2, X, y, groups[sel]

    ep_tr, Xtr, ytr, g_tr = prepare(epochs_train)
    _, Xte, yte, g_te = prepare(epochs_test)
    # test session(s) get distinct group ids so references never mix splits
    g_te = g_te + 1000

    band_list = list(bands) if bands else [None]
    sfreq = float(ep_tr.info["sfreq"])
    Z_tr_parts, Z_te_parts = [], []
    for band in band_list:
        if band is None:
            Xtr_b, Xte_b = Xtr, Xte
        else:
            from mne.filter import filter_data

            Xtr_b = filter_data(Xtr, sfreq, band[0], band[1], verbose=False, n_jobs=1)
            Xte_b = filter_data(Xte, sfreq, band[0], band[1], verbose=False, n_jobs=1)
        cov = Covariances(estimator=cov_est)
        Ctr = cov.fit_transform(Xtr_b)
        Cte = cov.transform(Xte_b)
        if align:
            refs_tr = _session_references(Ctr, g_tr, align)
            refs_te = _session_references(Cte, g_te, align)
            Ctr = _align_covs(Ctr, g_tr, refs_tr)
            Cte = _align_covs(Cte, g_te, refs_te)
        ts = TangentSpace(metric="riemann")
        Z_tr_parts.append(ts.fit_transform(Ctr))
        Z_te_parts.append(ts.transform(Cte))

    Ztr = np.concatenate(Z_tr_parts, axis=1)
    Zte = np.concatenate(Z_te_parts, axis=1)
    lr = LogisticRegression(max_iter=2000, solver="lbfgs")
    lr.fit(Ztr, ytr)
    return float(accuracy_score(yte, lr.predict(Zte)))

def erd_barplot(epochs, baseline=(-1.0, 0.0), active=(0.5, 2.0), use_csd=True):
    chs = epochs.ch_names
    if not set(['C3','C4']).issubset(chs):  # be forgiving (lowercase names, e.g., 'c3')
        lc = [c for c in chs if c.upper()=='C3']
        rc = [c for c in chs if c.upper()=='C4']
        if not (lc and rc): return
        c3, c4 = lc[0], rc[0]
    else:
        c3, c4 = 'C3','C4'
    def logpow(ep, ch, t0, t1):
        data = ep.copy().pick(ch).crop(tmin=t0, tmax=t1).get_data()
        return np.log(np.mean(data**2, axis=-1) + 1e-12).ravel()
    classes = pick_left_right(epochs)
    ep = epochs[classes].copy()
    ep = preprocess(ep, use_csd=use_csd)  # ensure bandpassed for ERD calc
    # Clamp baseline/active windows to available epoch time range
    ep_tmin, ep_tmax = float(ep.tmin), float(ep.tmax)
    bl0, bl1 = baseline
    ac0, ac1 = active
    s_bl0, s_bl1 = max(bl0, ep_tmin), min(bl1, ep_tmax)
    if s_bl1 <= s_bl0:
        span = max(0.25, min(0.5, ep_tmax - ep_tmin - 1e-3))
        s_bl0, s_bl1 = ep_tmin, ep_tmin + span
    s_ac0, s_ac1 = max(ac0, ep_tmin), min(ac1, ep_tmax)
    if s_ac1 <= s_ac0:
        span = max(0.25, min(0.5, ep_tmax - ep_tmin - 1e-3))
        s_ac0, s_ac1 = ep_tmax - span, ep_tmax
    bl_c3 = logpow(ep, c3, s_bl0, s_bl1); bl_c4 = logpow(ep, c4, s_bl0, s_bl1)
    ac_c3 = logpow(ep, c3, s_ac0, s_ac1); ac_c4 = logpow(ep, c4, s_ac0, s_ac1)
    erd_c3 = ac_c3 - bl_c3
    erd_c4 = ac_c4 - bl_c4
    y = ep.events[:, -1]
    id_map = {ep.event_id[classes[0]]:0, ep.event_id[classes[1]]:1}
    y = np.vectorize(id_map.get)(y)
    m = [erd_c3[y==0].mean(), erd_c4[y==0].mean(), erd_c3[y==1].mean(), erd_c4[y==1].mean()]
    plt.figure()
    plt.title(f"ERD (log power active-baseline) {classes[0]} vs {classes[1]}")
    plt.bar(['C3-'+classes[0], 'C4-'+classes[0], 'C3-'+classes[1], 'C4-'+classes[1]], m)
    plt.ylabel('Δ log power (8-30 Hz)')
    plt.tight_layout()

# ---- NEW: filter-bank relative bandpower + shrinkage LDA ----
def filterbank_relpow_lda(epochs, classes, active, baseline=(-1.0, 0.0),
                          bands=((8,12),(12,16),(16,20),(20,26),(26,30)),
                          use_laplacian=True, picks=None, groups=None, random_state=13):
    """Relative log-power per band/channel (active vs baseline) -> LDA(shrinkage='auto')."""
    ep = epochs[classes].copy().load_data()
    # Optional CSD (surface Laplacian)
    if use_laplacian and ep.get_montage() is not None:
        try:
            ep = mne.preprocessing.compute_current_source_density(ep)
        except Exception:
            pass
    # Optional channel selection
    if picks:
        keep = [ch for ch in picks if ch in ep.ch_names]
        if keep:
            ep.pick(keep)

    # Clamp windows to available epoch time range
    ep_tmin, ep_tmax = float(ep.tmin), float(ep.tmax)
    bl0, bl1 = baseline
    ac0, ac1 = active
    s_bl0, s_bl1 = max(bl0, ep_tmin), min(bl1, ep_tmax)
    if s_bl1 <= s_bl0:
        span = max(0.25, min(0.5, ep_tmax - ep_tmin - 1e-3))
        s_bl0, s_bl1 = ep_tmin, ep_tmin + span
    s_ac0, s_ac1 = max(ac0, ep_tmin), min(ac1, ep_tmax)
    if s_ac1 <= s_ac0:
        span = max(0.25, min(0.5, ep_tmax - ep_tmin - 1e-3))
        s_ac0, s_ac1 = ep_tmax - span, ep_tmax

    # Build features
    X_list = []
    for (lo, hi) in bands:
        ep_b = ep.copy().filter(lo, hi, fir_design='firwin', verbose=False)
        bl = ep_b.copy().crop(tmin=s_bl0, tmax=s_bl1).get_data()   # (n, ch, t)
        ac = ep_b.copy().crop(tmin=s_ac0, tmax=s_ac1).get_data()
        # relative log power: log(mean(ac^2)/mean(bl^2))
        rel = np.log(np.mean(ac**2, axis=-1) + 1e-12) - np.log(np.mean(bl**2, axis=-1) + 1e-12)
        # flatten channels
        X_list.append(rel.reshape(rel.shape[0], -1))
    X = np.concatenate(X_list, axis=1)
    # labels
    y = ep.events[:, -1]
    id_map = {ep.event_id[classes[0]]:0, ep.event_id[classes[1]]:1}
    y = np.vectorize(id_map.get)(y)

    # CV (grouped if provided)
    splitter = (GroupKFold(n_splits=5) if groups is not None
                else StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state))
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    acc = []
    for split in splitter.split(X, y, groups):
        tr, te = split
        lda.fit(X[tr], y[tr])
        acc.append(accuracy_score(y[te], lda.predict(X[te])))
    return np.array(acc)

# ---- NEW: optional time-window sweep helper ----
def sweep_windows(epochs, classes, windows, groups=None, use_laplacian=True, picks=None, bands=((8,12),(12,16),(16,20),(20,26),(26,30))):
    best = (-1, None)
    for (t0, t1) in windows:
        acc = filterbank_relpow_lda(epochs, classes, active=(t0, t1), groups=groups, use_laplacian=use_laplacian, picks=picks, bands=bands)
        m = acc.mean()
        if m > best[0]:
            best = (m, (t0, t1))
        print(f"FB-LDA {t0:.2f}-{t1:.2f}s  mean={m:.3f}  std={acc.std():.3f}")
    print(f"BEST FB-LDA window: {best[1][0]:.2f}-{best[1][1]:.2f}s  mean={best[0]:.3f}")
    return best

# ---- NEW: parse window grid spec ----
def parse_window_grid(spec):
    """Parse a spec like "starts=0.3:1.2:0.1,lengths=0.6,0.8,1.0,1.2,1.4" -> [(tmin,tmax), ...]."""
    if not spec:
        raise ValueError("Empty window grid spec")
    s = spec.replace(' ', '')
    if 'starts=' not in s or 'lengths=' not in s:
        raise ValueError("Spec must contain starts= and lengths=")
    # Split robustly without losing commas inside lengths
    after_starts = s.split('starts=')[1]
    if 'lengths=' not in after_starts:
        raise ValueError("Spec missing lengths=")
    starts_part, lengths_part = after_starts.split('lengths=')
    starts_part = starts_part.strip(',')
    lengths_part = lengths_part.strip(',')
    # starts: either a:b:step or comma list
    if ':' in starts_part:
        a, b, step = map(float, starts_part.split(':'))
        starts = list(np.arange(a, b + 1e-9, step))
    else:
        starts = [float(x) for x in starts_part.split(',') if x]
    lengths = [float(x) for x in lengths_part.split(',') if x]
    windows = [(round(st, 3), round(st + L, 3)) for st in starts for L in lengths]
    return windows

# ---- NEW: FB-LDA feature builder (no CV) ----
def fb_relpow_features(epochs, classes, active, baseline=(-1.0, 0.0),
                       bands=((8,12),(12,16),(16,20),(20,26),(26,30)),
                       use_laplacian=True, picks=None):
    ep = epochs[classes].copy().load_data()
    if use_laplacian and ep.get_montage() is not None:
        try:
            ep = mne.preprocessing.compute_current_source_density(ep)
        except Exception:
            pass
    if picks:
        keep = [ch for ch in picks if ch in ep.ch_names]
        if keep:
            ep.pick(keep)
    ep_tmin, ep_tmax = float(ep.tmin), float(ep.tmax)
    bl0, bl1 = baseline
    ac0, ac1 = active
    s_bl0, s_bl1 = max(bl0, ep_tmin), min(bl1, ep_tmax)
    s_ac0, s_ac1 = max(ac0, ep_tmin), min(ac1, ep_tmax)
    if s_bl1 <= s_bl0 or s_ac1 <= s_ac0:
        raise ValueError("Active/baseline window outside epoch range")
    X_list = []
    for (lo, hi) in bands:
        ep_b = ep.copy().filter(lo, hi, fir_design='firwin', verbose=False)
        bl = ep_b.copy().crop(tmin=s_bl0, tmax=s_bl1).get_data()
        ac = ep_b.copy().crop(tmin=s_ac0, tmax=s_ac1).get_data()
        rel = np.log(np.mean(ac**2, axis=-1) + 1e-12) - np.log(np.mean(bl**2, axis=-1) + 1e-12)
        X_list.append(rel.reshape(rel.shape[0], -1))
    X = np.concatenate(X_list, axis=1)
    y = ep.events[:, -1]
    id_map = {ep.event_id[classes[0]]:0, ep.event_id[classes[1]]:1}
    y = np.vectorize(id_map.get)(y)
    return X, y

# ---- NEW: nested CV for TS+LR ----
def nested_cv_ts_lr(epochs, classes, windows, cov_est='lwf', groups=None,
                    outer_splits=5, inner_splits=3, random_state=13):
    # prepare outer splitter
    # build y once using first window
    _, X0, y = crop_and_get_Xy(epochs, classes, windows[0][0], windows[0][1])
    if groups is not None:
        outer = GroupKFold(n_splits=outer_splits)
        outer_iter = outer.split(X0, y, groups)
    else:
        outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
        outer_iter = outer.split(X0, y)
    accs, chosen = [], []
    for out_idx, (tr, te) in enumerate(outer_iter):
        # Precompute X for all windows (all trials) then index per fold
        X_by_w = {}
        for (t0, t1) in windows:
            _, Xw, yw = crop_and_get_Xy(epochs, classes, t0, t1)
            X_by_w[(t0, t1)] = Xw
        # inner selection on training set
        best_m, best_w = -1.0, None
        inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
        for w in windows:
            Xtr = X_by_w[w][tr]; ytr = y[tr]
            # inner CV using the same pipeline as TS+LR
            fold_scores = []
            for tr2, va2 in inner.split(Xtr, ytr):
                C = Covariances(estimator=cov_est)
                TS = TangentSpace(metric='riemann')
                LR = LogisticRegression(max_iter=2000, solver='lbfgs')
                Ctr = C.fit_transform(Xtr[tr2]); Ztr = TS.fit_transform(Ctr)
                LR.fit(Ztr, ytr[tr2])
                Cva = C.transform(Xtr[va2]); Zva = TS.transform(Cva)
                fold_scores.append(accuracy_score(ytr[va2], LR.predict(Zva)))
            m = float(np.mean(fold_scores))
            if m > best_m:
                best_m, best_w = m, w
        # train on full train with best window, evaluate on test
        C = Covariances(estimator=cov_est)
        TS = TangentSpace(metric='riemann')
        LR = LogisticRegression(max_iter=2000, solver='lbfgs')
        Xtr_full = X_by_w[best_w][tr]; ytr_full = y[tr]
        Xte_full = X_by_w[best_w][te]; yte_full = y[te]
        Ctr = C.fit_transform(Xtr_full); Ztr = TS.fit_transform(Ctr)
        LR.fit(Ztr, ytr_full)
        Cte = C.transform(Xte_full); Zte = TS.transform(Cte)
        acc = accuracy_score(yte_full, LR.predict(Zte))
        accs.append(acc); chosen.append(best_w)
        print(f"[TS+LR][Outer {out_idx+1}] best_window={best_w[0]:.2f}-{best_w[1]:.2f}s  acc={acc:.3f}")
    return np.array(accs), chosen

# ---- NEW: nested CV for FB-LDA ----
def nested_cv_fb_lda(epochs, classes, windows, baseline=(-1.0,0.0),
                     bands=((8,12),(12,16),(16,20),(20,26),(26,30)),
                     use_laplacian=True, picks=None, groups=None,
                     outer_splits=5, inner_splits=3, random_state=13):
    # Build y from first window's features
    X0, y = fb_relpow_features(epochs, classes, active=windows[0], baseline=baseline,
                               bands=bands, use_laplacian=use_laplacian, picks=picks)
    if groups is not None:
        outer = GroupKFold(n_splits=outer_splits)
        outer_iter = outer.split(X0, y, groups)
    else:
        outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
        outer_iter = outer.split(X0, y)
    accs, chosen = [], []
    for out_idx, (tr, te) in enumerate(outer_iter):
        # Precompute features per window (all trials)
        X_by_w = {w: fb_relpow_features(epochs, classes, active=w, baseline=baseline,
                                        bands=bands, use_laplacian=use_laplacian, picks=picks)[0]
                  for w in windows}
        best_m, best_w = -1.0, None
        inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state)
        for w in windows:
            X = X_by_w[w]
            Xtr, ytr = X[tr], y[tr]
            fold_scores = []
            for tr2, va2 in inner.split(Xtr, ytr):
                lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
                lda.fit(Xtr[tr2], ytr[tr2])
                fold_scores.append(accuracy_score(ytr[va2], lda.predict(Xtr[va2])))
            m = float(np.mean(fold_scores))
            if m > best_m:
                best_m, best_w = m, w
        # Fit on full train, eval on test
        lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        Xtr_full = X_by_w[best_w][tr]; ytr_full = y[tr]
        Xte_full = X_by_w[best_w][te]; yte_full = y[te]
        lda.fit(Xtr_full, ytr_full)
        acc = accuracy_score(yte_full, lda.predict(Xte_full))
        accs.append(acc); chosen.append(best_w)
        print(f"[FB-LDA][Outer {out_idx+1}] best_window={best_w[0]:.2f}-{best_w[1]:.2f}s  acc={acc:.3f}")
    return np.array(accs), chosen

def main():
    ap = argparse.ArgumentParser(description="Riemannian MI sanity check (MDM & TS+LR) + ERD plot")
    ap.add_argument("epochs_fif", nargs='?', help="Path to epochs .fif or raw _online_raw.fif")
    ap.add_argument("--tmin", type=float, default=0.5)
    ap.add_argument("--tmax", type=float, default=2.0)
    ap.add_argument("--subject", type=str, help="Subject ID to aggregate all runs")
    ap.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / 'data', help="Directory containing sub-*_run-*_online_raw.fif")
    ap.add_argument("--runs", nargs='+', default=None, help="Optional run IDs to include (e.g. 01 02)")
    ap.add_argument("--epoch-tmin", type=float, default=-1.0, help="Epoch window start relative to event")
    ap.add_argument("--epoch-tmax", type=float, default=2.5, help="Epoch window end relative to event")
    ap.add_argument("--group-by-run", action="store_true", help="Use GroupKFold by run if metadata present")
    ap.add_argument("--sweep", action="store_true", help="Sweep a few decision windows with FB-LDA")
    ap.add_argument("--no-csd", action="store_true", help="Disable surface Laplacian/CSD (use plain average reference)")
    ap.add_argument("--picks", type=str, default="", help="Comma-separated channel list for FB-LDA (e.g., C3,Cz,C4)")
    ap.add_argument("--bands", type=str, default="8-12,12-16,16-20,20-26,26-30", help="FB-LDA bands as 'lo-hi,lo-hi,...'")
    ap.add_argument("--cov-est", type=str, choices=["lwf","oas"], default="lwf", help="Covariance estimator for Riemann models")
    ap.add_argument("--window-grid", type=str, default="", help="Grid spec 'starts=a:b:step,lengths=l1,l2,...' for window search")
    ap.add_argument("--nested", action="store_true", help="Run nested CV using the window grid (outer=5, inner=3)")
    ap.add_argument("--inner-splits", type=int, default=3, help="Inner CV splits for nested selection")
    args = ap.parse_args()

    use_csd = not args.no_csd
    bands = tuple(tuple(map(float, b.split('-'))) for b in args.bands.split(',') if '-' in b)
    picks = [p.strip() for p in args.picks.split(',') if p.strip()] or None

    grid_windows = None
    if args.window_grid:
        try:
            grid_windows = parse_window_grid(args.window_grid)
        except Exception as e:
            raise SystemExit(f"Failed to parse --window-grid: {e}")

    if args.subject:
        epochs = load_epochs_for_subject(
            args.subject,
            args.data_dir,
            epoch_tmin=args.epoch_tmin,
            epoch_tmax=args.epoch_tmax,
            runs=args.runs,
        )
    else:
        if not args.epochs_fif:
            raise SystemExit("Provide either --subject or a path to epochs_fif/raw_fif")
        epochs = load_epochs_from_input(args.epochs_fif, epoch_tmin=args.epoch_tmin, epoch_tmax=args.epoch_tmax)
    classes = pick_left_right(epochs)
    print(f"Using classes: {classes}")
    epochs = preprocess(epochs, use_csd=use_csd)
    groups = None
    if args.group_by_run and epochs.metadata is not None and "run" in epochs.metadata:
        groups = epochs.metadata["run"].to_numpy()
    ep, X, y = crop_and_get_Xy(epochs, classes, args.tmin, args.tmax)

    # Covariance-based pipelines
    acc_mdm, acc_ts = cv_scores_cov_models(X, y, cov_est=args.cov_est)
    print(f"MDM (Riemann)     : mean={acc_mdm.mean():.3f}  std={acc_mdm.std():.3f}")
    print(f"TangentSpace+LR   : mean={acc_ts .mean():.3f}  std={acc_ts .std():.3f}")

    # Label-shuffle control
    y_shuf = skshuffle(y, random_state=7)
    acc_mdm_s, acc_ts_s = cv_scores_cov_models(X, y_shuf, cov_est=args.cov_est, random_state=99)
    print(f"[Shuffle] MDM     : mean={acc_mdm_s.mean():.3f}")
    print(f"[Shuffle] TS+LR   : mean={acc_ts_s .mean():.3f}")

    # Quick ERD sanity plot if C3/C4 exist
    try:
        erd_barplot(ep, baseline=(-1.0, 0.0), active=(args.tmin, args.tmax), use_csd=use_csd)
    except Exception as e:
        print(f"ERD plot skipped: {e}")

    # Unified FB-LDA sweep: single-window by default, grid when --sweep
    windows = [(args.tmin, args.tmax)]
    if args.sweep:
        windows = [(0.4,1.2),(0.5,1.5),(0.6,1.6),(0.7,1.7),(0.8,1.8),(1.0,2.0)]
    best = sweep_windows(epochs, classes, windows, groups=groups, use_laplacian=use_csd, picks=picks, bands=bands)

    if args.nested:
        if not grid_windows:
            raise SystemExit("--nested requires --window-grid to be set")
        # Determine groups for outer CV
        outer_groups = groups if groups is not None else None
        print("\n=== Nested CV: TS+LR over window grid ===")
        ts_accs, ts_chosen = nested_cv_ts_lr(epochs, classes, grid_windows, cov_est=args.cov_est,
                                             groups=outer_groups, outer_splits=5, inner_splits=args.inner_splits)
        print(f"TS+LR nested mean={ts_accs.mean():.3f}  std={ts_accs.std():.3f}")
        # FB-LDA uses picks/bands and Laplacian toggle
        print("\n=== Nested CV: FB-LDA over window grid ===")
        fb_accs, fb_chosen = nested_cv_fb_lda(epochs, classes, grid_windows, baseline=(-1.0,0.0),
                                              bands=bands, use_laplacian=use_csd, picks=picks,
                                              groups=outer_groups, outer_splits=5, inner_splits=args.inner_splits)
        print(f"FB-LDA nested mean={fb_accs.mean():.3f}  std={fb_accs.std():.3f}")

    plt.show()

if __name__ == "__main__":
    main()