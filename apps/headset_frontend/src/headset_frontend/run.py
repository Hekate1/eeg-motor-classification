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
    seconds_per_epoch: float = 0.6,
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
    events = mne.find_events(raw, stim_channel="STI 014", shortest_event=1)
    X, y = preprocess_epoch_data(raw, events, EPOCH_TMIN, EPOCH_TMAX)
    y = (y - 1).astype(int)  # 1/2 -> 0/1

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

        title.draw()
        hint.draw()
        bars.update([float(proba[0]), float(proba[1])])
        bars.draw()
        win.flip()
        core.wait(seconds_per_epoch)

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
    parser.add_argument("--demo-seconds-per-epoch", type=float, default=0.6, help="Seconds to display each epoch.")
    args, _unknown = parser.parse_known_args(argv)

    if args.demo_fif is not None:
        _run_demo_fif(
            args.demo_fif,
            n_cal=args.demo_n_cal,
            seconds_per_epoch=args.demo_seconds_per_epoch,
        )
        return

    _run_hardware(argv)


if __name__ == "__main__":
    main()

