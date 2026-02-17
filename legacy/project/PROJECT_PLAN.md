## Project Implementation Plan: Data- and Domain-Efficient Motor Imagery Decoding

### 1. Current State Summary

- **Public dataset model (`model/src`)**
  - `run.py` trains a **Hybrid CNN + Transformer/Mamba** classifier (`HybridModelClassifier` in `hybrid_cnn_transformer.py`) on the PhysioNet/BCI-style dataset using:
    - Preprocessing + subject loaders in `data_utils.py`.
    - CSP-based feature modules (`CSPModule`, optional `SpectrogramModule` in `feature_modules.py`).
    - Rich augmentation strategies (`data_augmentation.py`) and cross-subject pretraining / subject-specific fine-tuning logic.
  - `visualization.py` already produces **training curves**, **strategy comparisons**, and **cross-subject heatmaps**, saving them under a `plots/` directory.
  - Classical baselines (`classical_baseline.py`) and parameter sweeps (`parameter_sweep.py`) are available but not yet tightly integrated into the final project narrative.

- **Self-collected EEG experiments (`experiment/`)**
  - **Data quality + sanity checks**:
    - `evaluate_mi_data.py` provides per-run sanity plots (PSD, ERPs, event counts) for raw FIF files recorded with the consumer headset.
    - `mi_riemann_probe.py` implements **Riemannian / filter-bank LDA pipelines** (MDM, TS+LR, FB-LDA, ERD plots) with label-shuffle controls and time-window sweeps, showing that there *is* motor imagery signal.
  - **Offline model training**:
    - `train_offline_model.py` loads epoch data from `sub-XX_run-YY_online_raw.fif`, applies a consistent preprocessing/epoching pipeline (via `online_preprocessing.py` and `online_mi_pipeline.py`), and
      - can train **simple CSP+sklearn models** (`--simple-model {rf,svm}`), or
      - trains a **subject-level `HybridModelClassifier` from scratch** with CSP features (`CSPModule`) and saves the model and basic history.
  - **Online pipeline (for potential real-time feedback)**:
    - `online_mi_pipeline.py`, `online_preprocessing.py`, `bar_feedback.py`, and `gamified_feedback.py` provide real-time epoching and feedback but are not yet driven by a well-validated, transfer-learned model.

- **Project notebooks (`project/project_proposal.ipynb`, `project/project.ipynb`)**
  - Contain the **proposal** and a graded **final-project skeleton** (abstract, intro, methods, implementation, discussion), but currently lack a clear, stepwise mapping from the existing codebase to the final results and figures.

---

### 2. High-Level Goals

1. **Strengthen and document the public-dataset baseline**
   - Ensure reproducible pretraining and fine-tuning on the public dataset, with clear training curves and cross-subject results.
   - Decide on one or two “canonical” backbone configurations (Transformer vs Mamba) and a small set of augmentation strategies.

2. **Build a robust classical baseline on self data**
   - Use `mi_riemann_probe.py` to confirm and quantify motor imagery signal with Riemannian/FB-LDA methods, including shuffled-label controls.
   - Select well-justified **time windows**, **frequency bands**, and **channel subsets** that work reliably for your headset.

3. **Implement and evaluate transfer learning from public dataset → self data**
   - Reuse the pretrained `HybridModelClassifier` weights from `model/src` with your own offline FIF data in `experiment/`.
   - Compare **from-scratch deep**, **transfer-learned deep**, and **classical** pipelines on the same epochs.

4. **Produce clear graphical evidence of progress**
   - Curate a **small, coherent set of figures** that show:
     - Classical baselines on public vs self data.
     - Deep model performance on public data (pretraining) vs self data (fine-tuning).
     - Ablation / augmentation strategy comparisons.
     - At least one ERD/ERSP-style visualization and one cross-subject/within-subject comparison plot.
   - Integrate these figures and brief quantitative summaries into `project/project.ipynb`.

---

### 3. Step-by-Step Plan

#### 3.1. Environment and Data Organization

1. **Lock down environments**
   - Reuse the existing `.venv` from `LIINC/.venv`.
   - Document any extra dependencies (e.g., `pyriemann`, `tqdm`, `seaborn`) explicitly in a requirements file or in the "Computational Methods" section of `project/project.ipynb`.

2. **Verify data locations & naming conventions**
   - Confirm that public dataset preprocessed FIFs live under `model/processed_data/` (or equivalent paths used by `data_utils.load_subject_data`).
   - Confirm that your self-collected runs use the `sub-XX_run-YY_online_raw.fif` pattern under `experiment/data/` (required by `train_offline_model.py` and `mi_riemann_probe.py`).

3. **Create a results directory hierarchy**
   - Standardize on something like:
     - `model/plots/` – training curves, cross-subject plots for public data (already used by `visualization.py`).
     - `experiment/plots/` – ERD/ERSP, Riemannian baselines, self-data training curves.
     - `experiment/results/` – CSV summaries of accuracies and hyperparameters for self data.
   - This will make it easy to track progress over time and pull figures into your final notebook.

---

#### 3.2. Public Dataset: Consolidate and Document the Baseline

1. **Run and log the current best public-data pipeline**
   - Use `model/src/run.py` to:
     - Train a base model on ~90 training subjects (`--pretrain`) and selectively fine-tune on held-out test subjects (`--finetune`), matching your proposal.
     - Evaluate classical Riemannian baselines with `--classical` and baseline performance with `--baseline`.
   - Ensure that `visualization.plot_base_model_results`, `plot_strategy_comparison`, and `visualize_cross_subject_results` are producing and saving correctly labelled PNGs under `model/plots/`.

2. **Pick a canonical configuration**
   - Decide on **one backbone and hyperparameter set** to carry forward (e.g., Transformer with the current `model_spec` in `run.py`).
   - Optionally, use the existing `--basesweep` and `--aug-sweep` flags to do a *lightweight* hyperparameter/augmentation sweep, saving the printed summary tables for inclusion in the project notebook.

3. **Summarize results in the notebook**
   - In `project/project.ipynb`, add a section that:
     - Shows the best training/validation curves and cross-subject strategy heatmap from `model/src/visualization.py`.
     - Reports a small table with **mean ± std** accuracy across test subjects for:
       - Classical Riemannian baseline.
       - Pretrained hybrid model without fine-tuning.
       - Pretrained + subject-finetuned model.

---

#### 3.3. Self Data: Classical Baseline and Sanity Checks

1. **Use `evaluate_mi_data.py` to qualitatively inspect runs**
   - For each self subject/run (e.g., `sub-01_run-01_online_raw.fif`, etc.):
     - Run `evaluate_mi_data.py` with `--no-browser` when scripting and optionally with ERPs (`--erp`) for at least one representative run.
     - Confirm clean PSDs (no dominant line noise), reasonable event distributions, and plausible ERPs.
   - Save a *small* subset of PSD/ERP figures (1–2 plots) under `experiment/plots/` for inclusion as “data quality” evidence in the final report.

2. **Quantify signal with Riemannian / FB-LDA baselines**
   - Use `mi_riemann_probe.py` to:
     - Run MDM and TS+LR at a default time window (e.g., `--tmin 0.5 --tmax 2.0`) and inspect accuracy and shuffle controls.
     - Use `--sweep` to explore several time windows and pick the best-performing one.
     - Optionally, use `--window-grid` and `--nested` to run nested CV for time-window selection.
   - Record for each subject:
     - Best TS+LR and FB-LDA accuracy (with mean ± std over folds).
     - Shuffle accuracies (should be near chance).
   - Save at least:
     - One **ERD bar plot** (C3/C4, Left vs Right) from `erd_barplot`.
     - One summary bar or boxplot of FB-LDA accuracy across runs.

3. **Decide on fixed analysis settings for self data**
   - From the sweeps above, choose:
     - A **fixed time window** (or very small set) for self data (e.g., 0.5–2.0s after cue).
     - A set of **filter bands** and **channels** (e.g., mu/beta 8–30 Hz, with or without Laplacian, C3/Cz/C4) that yields robust classical performance.
   - These settings will then be mirrored in the deep-learning pipeline as closely as possible.

---

#### 3.4. Self Data: Deep Model From Scratch vs Transfer Learning

1. **Standardize epoching and time length for self data**
   - Verify that `train_offline_model.py` currently crops/truncates all self-data trials to a common `common_length` divisible by 32.
   - Check the **time length used by the public-data model** (e.g., via a quick `load_subject_data` call in a small script or notebook) and, if needed, adjust `common_length` or the epoch window so that self-data `n_times` matches the base model’s expectation.
   - Document this alignment choice in `PROJECT_PLAN.md` and `project/project.ipynb`.

2. **Run the deep model from scratch on self data**
   - Use the current `train_offline_model.py` without any transfer:
     - Train the Hybrid model with the default hyperparameters (embedding dim, heads, layers, CSP components).
     - Optionally, also run `--simple-model rf` and/or `--simple-model svm` for CSP+sklearn baselines on the same epochs.
   - Save:
     - Training/validation accuracy curves (you can reuse the plotting style from `model/src/visualization.py` or add a light wrapper around `results['history']`).
     - Final validation accuracy along with classical FB-LDA accuracy for easy comparison.

3. **Add transfer learning support to `train_offline_model.py`**
   - **Code changes (high level):**
     - Extend the CLI with a flag like `--base-model-path` pointing to a pretrained public-data model (e.g., `model/src/models/base_model_90_subjects.pt`).
     - In `train_deep_model`:
       - If `args.base_model_path` is provided:
         - Load the pretrained classifier and metadata via `HybridModelClassifier.load_model`.
         - Replace the freshly instantiated classifier with this loaded instance (including its normalizer).
         - Rebuild or adapt CSP feature modules for the self dataset, ensuring their configuration (bands, `n_components`, `sfreq`) is compatible.
         - Optionally call `clf.freeze_layers(unfrozen_layers=[...])` (following `run.py`’s `finetune_subject_specific`) to only fine-tune lightweight heads and fusion layers.
       - Call `clf.train_and_evaluate(..., fine_tuning=True, use_lr_scheduler='onecycle')` so that the classifier respects the existing weights and normalization.
     - Ensure the saved model’s metadata clearly labels it as a **transfer-finetuned** model on consumer EEG (e.g., add a `source_model` field and a `dataset` field in `metadata`).

4. **Evaluate transfer vs scratch**
   - For each self subject (or for your main subject):
     - Run `train_offline_model.py` in **scratch** mode and **transfer** mode with identical epoching and CSP settings.
     - Collect: final validation accuracy, confusion matrices, and (if feasible) a small ablation (with/without CSP features) using the `HybridModelClassifier` machinery.
   - Summarize in a table:
     - FB-LDA accuracy.
     - Hybrid (scratch) accuracy.
     - Hybrid (transfer) accuracy with 95% CIs or std across folds.
   - Save at least one **side-by-side plot** (e.g., bar plot or line plot over increasing train set size) showing deep model improvement due to transfer vs scratch.

---

#### 3.5. Progress Visualization and Final Reporting

1. **Centralize metrics into CSVs**
   - For self data, create a simple writer utility (or just small scripts) that appends rows to a CSV under `experiment/results/`, with columns like:
     - `timestamp`, `subject`, `model_type` (FB-LDA, Hybrid-scratch, Hybrid-transfer, etc.), `window`, `bands`, `n_trials_train`, `mean_acc`, `std_acc`, `shuffle_acc` (if applicable), `notes`.
   - This CSV becomes the basis for aggregate plots and makes progress over time easy to visualize.

2. **Curate a minimal figure set**
   - **Data quality**: 1–2 panels from `evaluate_mi_data.py` (PSD, ERP or event-time plot) showing your headset recordings are reasonable.
   - **Public data**: 2–3 panels from `model/src/visualization.py`:
     - Base-model training curves.
     - Cross-subject augmentation strategy comparison / heatmap.
   - **Self data – classical vs deep**:
     - ERD bar plot + one accuracy bar chart for Riemannian/FB-LDA pipelines.
     - Training curves for scratch vs transfer; possibly overlayed or side-by-side.
     - A summary bar chart comparing FB-LDA, Hybrid-scratch, Hybrid-transfer.

3. **Integrate everything into `project/project.ipynb`**
   - Fill in the **Abstract, Introduction, Methods, Implementation, and Discussion** sections with:
     - Clear references to the code modules you used/extended (`run.py`, `train_offline_model.py`, `mi_riemann_probe.py`, `visualization.py`, etc.).
     - Short code or pseudocode snippets (not full files) that show how you called training/evaluation functions.
     - Embedded figures and tables generated in the steps above.
   - Emphasize in the Discussion:
     - Whether transfer learning meaningfully improved over scratch training and over classical methods.
     - How much labeled data you needed to reach above-chance performance on your consumer headset.
     - Limitations (small-N, headset noise, domain mismatch) and concrete next steps.

4. **Optional: online-feedback demo (stretch goal)**
   - If time permits, integrate the best-performing self-data model back into the **online** pipeline (`online_mi_pipeline.py`, `bar_feedback.py`, `gamified_feedback.py`):
     - Add a configuration path to load the transfer-finetuned `HybridModelClassifier` and its feature modules for real-time decoding.
     - Run a short online session and log predicted labels vs cues for qualitative analysis.
   - Capture a short screenshot or time-series plot showing online predictions and any bar/gamified feedback, and include it as a final qualitative result.

---

### 4. How This Plan Supports Graphical Progress Tracking

- **Early stage**: Figures from `evaluate_mi_data.py` and `mi_riemann_probe.py` document raw data quality and the existence of MI signal via simple methods.
- **Middle stage**: `model/src/visualization.py` plots and public-data baselines show that the hybrid model is well-behaved and high-performing in a standard setting.
- **Late stage**: New plots from `train_offline_model.py` (scratch vs transfer curves & bar charts) plus aggregated CSV summaries visualize how your self-data performance evolves as you move from classical → deep scratch → deep with transfer.
- These curated figures and tables can be dropped directly into `project/project.ipynb` to tell a clear, graphical story of progress from initial failures (chance-level deep model on small self data) to a more principled, data- and domain-efficient solution.