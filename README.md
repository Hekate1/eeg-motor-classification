# EEG Motor Imagery Classification (Headset App + Training Pipeline + Results)

![Demo replay of the online BCI frontend](docs/figures/demo_replay.gif)

*The PsychoPy frontend decoding recorded EEG in replay mode: the bars show the classifier's
left/right probability updating as each imagery epoch streams in. Runs with zero hardware:*
`eeg-headset-frontend --demo-fif data/self/sub-01_run-01_online_raw.fif`

This repo is an end-to-end motor-imagery EEG project with three runnable components:

1) **Headset-facing “frontend” app** (PsychoPy + BrainFlow) for online cueing + feedback  
2) **Offline processing + model training pipeline** (classical + deep learning; transfer learning supported)  
3) **Results notebook** that summarizes offline accuracies across settings (and saves plots/tables)

## Results at a glance

All results are 2-class (left vs. right hand) motor imagery for a single subject: ~770 trials
self-recorded over 8 sessions with a 16-channel OpenBCI Cyton + Daisy. **Chance is 50%.**
The deep model is a hybrid CNN–Transformer; “transfer” means it was pretrained on 90 subjects
from the [PhysioNet EEG Motor Movement/Imagery dataset](https://physionet.org/content/eegmmidb/1.0.0/)
and fine-tuned on the self-recorded data.

| Evaluation (both cross-session) | TS+LR + session alignment | Riemannian TS+LR | Transfer (pretrained) | Deep net from scratch |
|---|---|---|---|---|
| Leave-one-run-out (7 calibration runs) | **63% ± 7** | 62% ± 9 | 53% ± 6 | 52% ± 5 |
| 1 calibration run (~95 trials) | **57% ± 8** | 54% ± 6 | 50% ± 5 | 51% ± 4 |

Numbers are from a leakage-controlled protocol (single pinned environment,
`requirements/lock-gpu.txt`): checkpoint selection uses an inner validation split carved
from the *training* runs, each held-out run is evaluated exactly once, and deep results
average over 3 seeds.¹ Cross-session, the classical Riemannian tangent-space + logistic
regression pipelines are the only methods clearly above chance on this dataset; the deep
models — pretrained or not — do not generalize across sessions with this recipe, even
though transfer clearly helps *within* a session (59% vs 53% on a random split).

The best method recenters each session's trial covariances at that session's own mean
(Riemannian session alignment) before the tangent-space projection, adds filter-bank
sub-band features (8–12/12–16/16–22/22–30 Hz), and drops the four frontal-most channels.
Each choice was validated on paired train/test draws against the plain baseline
(+2.8 points overall, Wilcoxon p = 0.003, n = 56 paired configurations). Alignment needs
no labels from the new session, so it works online. Its biggest effect is where it matters
practically — the low-calibration end: **one aligned calibration run matches what the
unaligned baseline needs 2–4 runs to reach**, roughly halving setup time for a new session.

**Cross-session generalization** (train on 7 runs, test on the held-out 8th: the hard, honest
split for BCI, since session-to-session drift is the main failure mode):

![Leave-one-run-out generalization](docs/figures/loro_generalization.png)

**How much calibration data does a new session need?** For each held-out run, models were
trained on k runs drawn at random from the other 7 (two draws per test run, identical draws
for every method and seed, so comparisons are paired) and tested on the held-out run. The
classical pipelines improve steadily with calibration data (57% → 63% aligned); the deep
models stay near chance at every k, and the paired pretraining gain (transfer − scratch on
identical train/test splits) is zero within noise. A frozen-backbone / low-learning-rate
fine-tuning variant did no better; per-session Euclidean alignment of the raw trials lifts
transfer to 55.5% at LORO — a real (+1 pt paired) but modest effect that still trails the
classical pipeline.

![Cross-session accuracy vs. amount of calibration data](docs/figures/calibration_curve.png)

**Other findings from the improvement sweep** (840 paired fits, `split=improvement_v1` in
the metrics CSV):

- **The usable signal lives in 0.5–2.0 s post-cue.** These sessions hold 2.5 s of imagery
  per trial, but extending the analysis window to 2.5 s *costs* 4.9 points paired — class
  separability collapses in the final half-second, when the on-screen feedback ends.
- **Frontal channels hurt generalization.** Dropping Fp1/Fp2/F7/F8 helps (+1.1 paired)
  — consistent with the frontally-dominated CSP patterns below: ocular activity carries
  class-correlated variance within a session that does not survive session drift.
- **Session drift is partly physical.** μ-band ERD lateralization at C3/C4 varies strongly
  across sessions and even reverses sign in two of them (`scripts/diagnose_runs.py`),
  pointing at electrode repositioning between sessions as a major drift source — exactly
  the kind of spatial shift covariance alignment can partially absorb.
- A surface Laplacian (CSD) did not help (−1.2 paired).

¹ *Earlier versions of this repo reported higher cross-session numbers; those were affected
by issues in how accuracy was computed and have been superseded by the protocol above.*

**The signal is physiological:** μ-band (8–13 Hz) power over the right motor cortex (C4) is
suppressed during left-hand imagery — the classic contralateral event-related desynchronization
that motor-imagery BCIs are built on. The effect is weaker over C3, one reason single-session
accuracy tops out where it does.

![Class-conditional spectra at C3/C4](docs/figures/erd_spectra.png)

<details>
<summary><b>More figures:</b> confusion matrices & CSP spatial patterns</summary>

![Validation confusion matrices](docs/figures/confusion_matrices.png)

CSP spatial patterns (8–30 Hz), fit on all runs. The leading patterns are frontally
dominated, a sign that residual ocular/frontal activity still carries class-correlated
variance after preprocessing. That is a well-known caveat for motor-imagery BCIs and part
of the motivation for the run-wise evaluation above:

![CSP spatial patterns](docs/figures/csp_patterns.png)

</details>

All figures are regenerated from the committed data and metrics with:

```bash
python scripts/generate_report_figures.py
```

## Repo layout

```
apps/
  headset_frontend/           # hardware + demo replay frontend
pipeline/
  src/pipeline/public/        # public-dataset training scripts (expects processed epochs)
  src/pipeline/self/          # self-data training + classical baselines
notebooks/
  results_summary.ipynb       # report notebook (repo-relative paths)
data/
  self/                       # self-collected FIF runs (sanitized)
docs/
  figures/                    # report figures embedded above (regenerable)
scripts/
  sanitize_fif_metadata.py
  download_public_dataset.py
  generate_report_figures.py  # regenerates docs/figures/ from committed data
  run_calibration_sweep.py    # cross-session calibration experiment (v1, superseded)
  run_calibration_sweep_v2.py # cross-session calibration sweep behind the README numbers
  run_improvement_sweep.py    # paired sweep: alignment / filter bank / window / channels
  diagnose_runs.py            # per-session signal-quality diagnostics
configs/
  default.yaml
requirements/
  base.txt
  frontend.txt
  ml.txt
  lock-gpu.txt                # pinned environment behind the README numbers
legacy/                       # snapshot of the original project layout (for reference)
```

## Quickstart

### Install

This is a Python-only project. Start with a clean Python 3.10+ environment.

**Default (recommended): full install**

```bash
python -m pip install -e ".[all]"
```

If the full install fails (most commonly due to **PsychoPy/BrainFlow** system dependencies on Linux, or CUDA/PyTorch setup), you can install a smaller subset and still run most of the repo:

```bash
# Base (offline analysis + utilities)
python -m pip install -e ".[base]"

# Training + classical baselines (no headset app)
python -m pip install -e ".[base,ml]"

# Headset app only (no ML training stack)
python -m pip install -e ".[base,frontend]"
```

Notes:
- **PsychoPy** can require OS-level dependencies on Linux. If install fails, see the [PsychoPy website](https://www.psychopy.org/download.html).

### 1) Headset frontend (no hardware demo)

Replay a recorded run and drive the PsychoPy bar UI without a headset connected:

```bash
eeg-headset-frontend --demo-fif data/self/sub-01_run-01_online_raw.fif
```

The demo replays in **real time** by default. To speed up or slow down:

```bash
eeg-headset-frontend --demo-fif data/self/sub-01_run-01_online_raw.fif --demo-speed 2.0
```

### 1b) Record your own data (hardware)

This repo’s “frontend” is a PsychoPy + BrainFlow app designed for **OpenBCI Cyton + Daisy** (16 channels) recordings.

#### Prereqs
- A supported EEG device (tested with **OpenBCI Cyton + Daisy**)
- BrainFlow installed (`pip install -e ".[frontend]"`)
- On Linux you may need:
  - permissions to access the serial device (e.g. adding your user to `dialout`)
  - PsychoPy system dependencies (see PsychoPy docs)

#### Record a run (cueing + raw recording)

This writes an MNE Raw `.fif` containing EEG + a marker channel:
- Output naming: `sub-<ID>_run-<NN>_online_raw.fif`

Run from the repo root:

```bash
mkdir -p data/self
cd data/self
eeg-headset-frontend --subject 01 --run 1 --trials 120
```

Tips:
- Use a new `--run` number each session (it’s also used as the RNG seed for balanced cue ordering).
- If auto-detecting the serial port fails, pass `--port <device_path>` (see `--help`).
- Use `--game` for the gamified feedback view.

#### Sanitize before sharing

If you plan to publish your recordings, sanitize FIF metadata:

```bash
python scripts/sanitize_fif_metadata.py data/self --overwrite
```

### 2) Self-data pipeline (fast classical baseline)

Run a quick CSP + RandomForest baseline on the included self-recorded runs:

```bash
eeg-self-train --data-dir data/self --runs 01 02 03 04 05 06 --simple-model rf --model-name quick_rf
```

Outputs are written under `runs/self/` by default.

### 3) Results notebook

Open:
- `notebooks/results_summary.ipynb` ([view rendered on nbviewer](https://nbviewer.org/github/Hekate1/eeg-motor-classification/blob/main/notebooks/results_summary.ipynb))

The first setup cell finds the repo root automatically and uses:
- self-data: `data/self/`
- outputs: `runs/self/`

## Public dataset (download locally; not committed)

The public dataset is intentionally excluded from git (too large). The public training code (`eeg-public-train`) expects **preprocessed epochs** at:

- `data/public/processed_data/sub-XXX_run-R_processed-epo.fif`

You can generate these locally with `scripts/download_public_dataset.py`.

Set the environment variable so the training code can find your processed epochs:

```bash
export PUBLIC_PROCESSED_DATA_DIR=data/public/processed_data
```

### Option A: quick smoke test (small)

Useful to validate your environment and that the preprocessing is working:

```bash
python scripts/download_public_dataset.py --subjects 1 2 3 --runs 3 7 11
```

### Option B: full pretrain setup (recommended)

To match the default pretraining config in `pipeline/src/pipeline/public/run.py` (which trains a base model on **90 subjects**), you’ll want processed epochs for *many* subjects.

This downloads and preprocesses motor-imagery runs (3/7/11) for subjects 1–109:

```bash
python scripts/download_public_dataset.py --subjects $(seq 1 109) --runs 3 7 11
```

## Training the “best” model (public pretrain → transfer to your headset data)

This is the intended high-performance workflow:

1) **Preprocess** the public dataset into `data/public/processed_data` (Option B above).

2) **Pretrain** on many public subjects (compute-heavy). This writes:
- `models/base_model_<N>_subjects.pt` where \(N\) is the **number of subjects actually loaded**.

Note: the default `eeg-public-train` configuration *targets* 90 training subjects (and holds out 10 test subjects) from the 109 available. If you only preprocess a few subjects, \(N\) will be smaller and the checkpoint name will reflect that.

```bash
eeg-public-train --pretrain
```

3) **Fine-tune / transfer** to your self-collected runs:

Use either:
- the model you just trained at `models/base_model_<N>_subjects.pt`, or
- the included reference checkpoint at `legacy/model/models/base_model_90_subjects.pt`

```bash
eeg-self-train \
  --data-dir data/self \
  --runs 01 02 03 04 05 06 \
  --base-model-path models/base_model_<N>_subjects.pt \
  --model-name offline_transfer_custom
```

The result will be logged to `runs/self/results/offline_metrics.csv` and included in `notebooks/results_summary.ipynb`.

## Architecture (high level)

```mermaid
flowchart TD
  Headset[OpenBCI_CytonDaisy] --> Brainflow[BrainFlow_Stream]
  Brainflow --> Frontend[PsychoPy_Frontend]
  Brainflow --> Recorder[MNE_FIF_Recorder]
  Recorder --> SelfPipeline[SelfData_OfflinePipeline]
  PublicData[Public_MI_Dataset] --> PublicPipeline[PublicData_TrainingPipeline]
  PublicPipeline --> BaseModel[Pretrained_Model]
  BaseModel --> SelfPipeline
  SelfPipeline --> Metrics[Metrics_CSV_Plots]
  Metrics --> Notebook[Results_Summary_Notebook]
```

## Data and privacy notes

- **Self-collected `.fif` runs** are included under `data/self/`. Before inclusion, metadata sanitization is performed via:
  - `scripts/sanitize_fif_metadata.py`
- **Public dataset** and **derived public processed epochs** are excluded by `.gitignore` and are intended to be generated locally.

## Code sections

- **Headset app**: `apps/headset_frontend/src/headset_frontend/online_mi_pipeline.py`
- **Self-data training**: `pipeline/src/pipeline/self/train_offline_model.py`
- **Classical baseline**: `pipeline/src/pipeline/self/mi_riemann_probe.py`
- **Public training entrypoint**: `pipeline/src/pipeline/public/run.py` (wrapped by `pipeline/src/pipeline/public/train.py`)
- **Results notebook**: `notebooks/results_summary.ipynb`

## Limitations / future work

- **Session drift**: run-wise generalization remains challenging (nonstationarity).
- **Online inference**: integrating the best offline model into the online loop (latency + calibration) can be improved and validated more rigorously.

## What’s intentionally missing (for portability)

- The full **public dataset** payload is not included (too large). Use `scripts/download_public_dataset.py`.
- GPU/CUDA setup is not automated (varies by machine). If `torch` install is finicky, use `python -m pip install -e ".[base]"` or `".[base,frontend]"` and still run the demo + classical baselines.

