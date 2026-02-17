# Headset frontend app

This component contains the **headset-facing application**: PsychoPy for the UI and BrainFlow for OpenBCI streaming.

## Run (no hardware demo)

```bash
eeg-headset-frontend --demo-fif ../../data/self/sub-01_run-01_online_raw.fif
```

## Run (hardware)

```bash
eeg-headset-frontend --subject 01 --run 1 --trials 120
```

The hardware mode is implemented in `src/headset_frontend/online_mi_pipeline.py` and accepts additional flags (e.g., `--port`, `--game`).

