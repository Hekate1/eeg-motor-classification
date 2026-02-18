#!/usr/bin/env python3
"""
Entry point for the headset-facing app.

Modes:
- Hardware: run the online BrainFlow + PsychoPy pipeline (default).
- Demo: replay a recorded `.fif` file and drive the PsychoPy UI without hardware.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _run_hardware(argv: list[str] | None = None) -> None:
    # Defer heavy imports until actually needed.
    from . import online_mi_pipeline

    # `online_mi_pipeline.main()` parses `sys.argv`; it doesn't take argv.
    # So we just call it directly and let argparse handle the current CLI.
    online_mi_pipeline.main()


def _run_demo_fif(
    fif_path: Path,
    *,
    n_cal: int = 40,
    n_csp_components: int = 6,
    speed: float = 1.0,
    cue_seconds: float | None = None,
) -> None:
    import mne
    import numpy as np
    from psychopy import core, event, visual
    from mne.decoding import CSP
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler

    from .bar_feedback import BarFeedback
    from .online_mi_pipeline import EPOCH_TMIN, EPOCH_TMAX
    from .online_preprocessing import preprocess_epoch_data

    raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose=False)
    sfreq = float(raw.info["sfreq"])
    events = mne.find_events(raw, stim_channel="STI 014", shortest_event=1)

    # Preprocess + epoch for features/labels.
    X, y = preprocess_epoch_data(raw, events, EPOCH_TMIN, EPOCH_TMAX)
    y = (y - 1).astype(int)  # 1/2 -> 0/1

    # Recreate epochs (same preprocessing) to recover event timing after drops,
    # so the demo runs at the same pace as the original recording.
    raw_copy = raw.copy()
    raw_copy.notch_filter(freqs=60, picks="eeg", method="iir", verbose=False)
    raw_copy.filter(l_freq=1.0, h_freq=50.0, picks="eeg", method="iir", verbose=False)
    raw_copy.set_eeg_reference("average", verbose=False)
    epochs = mne.Epochs(
        raw_copy,
        events,
        event_id={"Left": 1, "Right": 2},
        tmin=EPOCH_TMIN,
        tmax=EPOCH_TMAX,
        baseline=None,
        proj=False,
        picks="eeg",
        preload=True,
        verbose=False,
        on_missing="ignore",
    )
    data = epochs.get_data()
    variances = np.var(data, axis=(1, 2))
    z_scores = np.abs((variances - np.mean(variances)) / (np.std(variances) + 1e-12))
    bad_idx = np.where(z_scores > 2.0)[0]
    if len(bad_idx) > 0:
        epochs.drop(bad_idx)

    onset_times = epochs.events[:, 0].astype(int) / sfreq
    if len(onset_times) != len(y):
        onset_times = np.arange(len(y)) * float(EPOCH_TMAX - EPOCH_TMIN)

    if len(y) < 10:
        raise ValueError(f"Not enough epochs found in {fif_path} (n={len(y)}).")

    n_cal = max(4, min(int(n_cal), len(y) - 1))

    # ---------------- train a lightweight CSP+SGD baseline ----------------
    csp = CSP(n_components=n_csp_components, reg=None, log=True, norm_trace=False)
    X_cal, y_cal = X[:n_cal], y[:n_cal]
    feats_cal = csp.fit_transform(X_cal, y_cal)

    scaler = StandardScaler()
    feats_cal = scaler.fit_transform(feats_cal)

    clf = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=2000, tol=1e-3, random_state=0)
    clf.fit(feats_cal, y_cal)

    # ---------------- PsychoPy UI loop ----------------
    win = visual.Window((900, 600), fullscr=False, color="white", units="pix")
    bars = BarFeedback(win)
    title = visual.TextStim(
        win,
        text=f"Demo replay: {fif_path.name}",
        pos=(0, 250),
        height=26,
        color="black",
    )
    hint = visual.TextStim(
        win,
        text="Press ESC to quit",
        pos=(0, 220),
        height=18,
        color="black",
    )
    cue = visual.TextStim(win, text="", pos=(0, 120), height=60, color="black")
    cue_sm = visual.TextStim(win, text="", pos=(0, 170), height=22, color="black")

    epoch_window_s = float(EPOCH_TMAX - EPOCH_TMIN)
    if cue_seconds is None:
        cue_seconds = epoch_window_s
    speed = max(1e-6, float(speed))
    step_s = 0.05  # UI update step; keeps demo smooth without busy-waiting
    epoch_n_times = int(X.shape[2])

    for i in range(n_cal, len(y)):
        if "escape" in event.getKeys():
            break

        feats = csp.transform(X[i : i + 1])
        feats = scaler.transform(feats)
        proba = clf.predict_proba(feats)[0]  # [P(left), P(right)]

        # Ensure numeric stability / shape
        proba = np.asarray(proba, dtype=float)
        proba = np.clip(proba, 0.0, 1.0)
        proba = proba / (proba.sum() + 1e-12)

        true_label = "Left" if int(y[i]) == 0 else "Right"
        pred_label = "Left" if float(proba[0]) >= float(proba[1]) else "Right"

        if i + 1 < len(onset_times):
            trial_duration = max(0.0, float(onset_times[i + 1] - onset_times[i]))
        else:
            trial_duration = epoch_window_s
        trial_duration = trial_duration / speed

        # --- draw cue + continuously-updated bars ---
        cue.text = true_label
        cue_sm.text = f"Pred: {pred_label}  |  True: {true_label}"
        bars.update([float(proba[0]), float(proba[1])])

        t_start = core.getTime()
        t_end = t_start + trial_duration
        last_proba = proba.copy()
        while core.getTime() < t_end:
            if "escape" in event.getKeys():
                t_end = -1
                break

            t_into = core.getTime() - t_start

            # During the imagination epoch window, update the probability as more
            # samples from the epoch "arrive", similar to online decoding.
            if t_into <= epoch_window_s:
                rel = int(round(t_into * sfreq))
                rel = max(0, min(rel, epoch_n_times - 1))
                seg = X[i, :, : rel + 1]  # (n_ch, k)
                k = int(seg.shape[1])
                if k < epoch_n_times:
                    pad = np.repeat(seg[:, :1], epoch_n_times - k, axis=1)
                    seg_fixed = np.concatenate([pad, seg], axis=1)
                else:
                    seg_fixed = seg[:, -epoch_n_times:]

                feats = csp.transform(seg_fixed[None, :, :])
                feats = scaler.transform(feats)
                last_proba = clf.predict_proba(feats)[0]

            # Always show the latest probability estimate
            bars.update([float(last_proba[0]), float(last_proba[1])])
            title.draw()
            hint.draw()
            bars.draw()

            if t_into <= float(min(cue_seconds, epoch_window_s)):
                cue.draw()
            cue_sm.text = (
                f"Pred: {'Left' if float(last_proba[0]) >= float(last_proba[1]) else 'Right'}"
                f"  |  True: {true_label}"
            )
            cue_sm.draw()
            win.flip()
            core.wait(min(step_s, max(0.0, t_end - core.getTime())))

        if t_end < 0:
            break

    win.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Headset frontend (hardware) or demo replay (no hardware).",
    )
    parser.add_argument(
        "--demo-fif",
        type=Path,
        default=None,
        help="Path to a recorded `sub-XX_run-YY_online_raw.fif` to replay without hardware.",
    )
    parser.add_argument("--demo-n-cal", type=int, default=40, help="Calibration epochs for the demo CSP+SGD model.")
    parser.add_argument("--demo-speed", type=float, default=1.0, help="Replay speed multiplier (1.0 = real time).")
    parser.add_argument(
        "--demo-cue-seconds",
        type=float,
        default=None,
        help="Seconds to display the cue per trial (default: full epoch window).",
    )
    args, _unknown = parser.parse_known_args(argv)

    if args.demo_fif is not None:
        _run_demo_fif(
            args.demo_fif,
            n_cal=args.demo_n_cal,
            speed=args.demo_speed,
            cue_seconds=args.demo_cue_seconds,
        )
        return

    _run_hardware(argv)


if __name__ == "__main__":
    main()

