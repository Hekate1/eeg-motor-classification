#!/usr/bin/env python3
import os, argparse, numpy as np, matplotlib.pyplot as plt, seaborn as sns, mne
from mne.time_frequency import psd_array_welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from mne.decoding import CSP
from enhanced_preprocessing import (
    load_subject_data as load_raw,
    notch_highpass_rereference,
    auto_reject_channels,
    run_ica,
    bandpass_filter,
    epoch_data,
    auto_reject_trials
)
from data_utils import load_subject_data as load_processed_data, load_multi_subject_data
from feature_modules import CSPModule, SpectrogramModule
from hybrid_cnn_transformer import HybridModelClassifier
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def bool_print(string):
    if True:
        print(string)

def main(subject, subjects, runs, out_dir, baseline_n=0, base_model_path=None):
    os.makedirs(out_dir, exist_ok=True)
    # Build list of target subjects for ablation (fall back to single subject)
    subject_list = subjects if subjects is not None else [subject]
    # Baseline group setup
    baseline_ids = []
    if baseline_n > 0:
        # select first N subjects excluding target subjects
        all_subj = [f"{i:03d}" for i in range(1,110) if f"{i:03d}" not in subject_list]
        baseline_ids = all_subj[:baseline_n]
    #     # initialize baseline correlation data
    #     baseline_corrs_by_run = {run: [] for run in runs}
    #     # raw baseline metrics
    #     baseline_vars_by_run = {run: [] for run in runs}
    #     baseline_psd_means_by_run = {run: [] for run in runs}
    #     baseline_bad_counts_by_run = {run: [] for run in runs}
    #     for bsubj in baseline_ids:
    #         for run in runs:
    #             braw = load_raw(f"EEG Motor Movement:Imagery Dataset/sub-{bsubj}/eeg/sub-{bsubj}_task-motion_run-{run}_eeg.set")
    #             bdata = braw.get_data()
    #             baseline_vars_by_run[run].extend(np.var(bdata, axis=1).tolist())
    #             bpsd, bfreqs = psd_array_welch(bdata, sfreq=braw.info['sfreq'], fmin=1, fmax=50, n_fft=2048)
    #             baseline_psd_means_by_run[run].append(bpsd.mean(axis=0))
    #             # compute baseline bads after preprocessing
    #             b1 = notch_highpass_rereference(braw)
    #             b2 = auto_reject_channels(b1, do_print=False)
    #             baseline_bad_counts_by_run[run].append(len(b2.info.get('bads', [])))
    #             # compute baseline correlation
    #             corr_b = np.corrcoef(braw.get_data())
    #             baseline_corrs_by_run[run].append(corr_b)
    #     # compute baseline summaries
    #     baseline_var_stats = {}
    #     baseline_psd_mean = {}
    #     baseline_psd_std = {}
    #     # compute baseline mean correlation matrices
    #     baseline_corr_mean = {}
    #     for run in runs:
    #         arr = np.array(baseline_vars_by_run[run])
    #         baseline_var_stats[run] = {'mean': arr.mean(), 'std': arr.std(), 'min': arr.min(), 'max': arr.max()}
    #         stack = np.vstack(baseline_psd_means_by_run[run])
    #         baseline_psd_mean[run] = stack.mean(axis=0)
    #         baseline_psd_std[run] = stack.std(axis=0)
    #         # compute mean correlation across baseline subjects for this run
    #         corr_stack = np.stack(baseline_corrs_by_run[run], axis=0)
    #         baseline_corr_mean[run] = corr_stack.mean(axis=0)

    # bads, before, after, Xs, ys = [], [], [], [], []
    # info_csp = None
    # for run in runs:
    #     raw = load_raw(f"EEG Motor Movement:Imagery Dataset/sub-{subject}/eeg/sub-{subject}_task-motion_run-{run}_eeg.set")
    #     # PSD (subject vs baseline)
    #     data_raw = raw.get_data()
    #     # bool_print raw data summary
    #     raw_min, raw_max = data_raw.min(), data_raw.max()
    #     raw_mean, raw_std = data_raw.mean(), data_raw.std()
    #     # bool_print(f"Run {run} raw data: mean {raw_mean:.3e}, std {raw_std:.3e}, min {raw_min:.3e}, max {raw_max:.3e}")
    #     psd, freqs = psd_array_welch(data_raw, sfreq=raw.info['sfreq'], fmin=1, fmax=50, n_fft=2048)
    #     sub_psd = psd.mean(axis=0)
    #     fig, ax = plt.subplots(figsize=(6,4))
    #     if baseline_n > 0:
    #         mean_b = baseline_psd_mean[run]
    #         std_b = baseline_psd_std[run]
    #         ax.semilogy(freqs, mean_b, color='gray', label='Baseline mean')
    #         ax.fill_between(freqs, mean_b-std_b, mean_b+std_b, color='gray', alpha=0.3)
    #     ax.semilogy(freqs, sub_psd, color='blue', label=f'Sub {subject}')
    #     ax.set_xlim(1, 50)
    #     ax.set_xlabel('Frequency (Hz)')
    #     ax.set_ylabel('PSD')
    #     ax.set_title(f'Subject {subject} Run {run} - PSD')
    #     ax.legend()
    #     plt.savefig(f"{out_dir}/psd_{run}.png")
    #     # bool_print PSD summary
    #     sub_min_psd, sub_max_psd = sub_psd.min(), sub_psd.max()
    #     sub_mean_psd = sub_psd.mean()
    #     if baseline_n > 0:
    #         base_min_psd, base_max_psd = mean_b.min(), mean_b.max()
    #         base_mean_psd = mean_b.mean()
    #         bool_print(f"Run {run} PSD (1-50Hz): subject mean {sub_mean_psd:.3e}, min {sub_min_psd:.3e}, max {sub_max_psd:.3e}; baseline mean {base_mean_psd:.3e}, min {base_min_psd:.3e}, max {base_max_psd:.3e}")
    #     else:
    #         bool_print(f"Run {run} PSD (1-50Hz): subject mean {sub_mean_psd:.3e}, min {sub_min_psd:.3e}, max {sub_max_psd:.3e}")
    #     # Channel variance distribution (subject vs baseline)
    #     var = np.var(data_raw, axis=1)
    #     fig, ax = plt.subplots(figsize=(6,4))
    #     if baseline_n > 0:
    #         sns.histplot(baseline_vars_by_run[run], bins=50, color='gray', stat='density', alpha=0.5, label='Baseline', ax=ax)
    #     sns.histplot(var, bins=50, color='blue', stat='density', alpha=0.7, label=f'Sub {subject}', ax=ax)
    #     ax.set_title(f'Subject {subject} Run {run} - Channel Variance')
    #     ax.set_xlabel('Variance')
    #     ax.legend()
    #     plt.savefig(f"{out_dir}/var_{run}.png")
    #     # bool_print variance summary
    #     var_min, var_max = var.min(), var.max()
    #     var_mean = var.mean()
    #     if baseline_n > 0:
    #         base_var_arr = np.array(baseline_vars_by_run[run])
    #         base_var_min, base_var_max = base_var_arr.min(), base_var_arr.max()
    #         base_var_mean = base_var_arr.mean()
    #         bool_print(f"Run {run} variance: subject mean {var_mean:.3e}, min {var_min:.3e}, max {var_max:.3e}; baseline mean {base_var_mean:.3e}, min {base_var_min:.3e}, max {base_var_max:.3e}")
    #     else:
    #         bool_print(f"Run {run} variance: subject mean {var_mean:.3e}, min {var_min:.3e}, max {var_max:.3e}")
    #     # Channel correlation
    #     if baseline_n > 0 and 'baseline_corr_mean' in locals():
    #         fig, axes = plt.subplots(1,2,figsize=(12,4))
    #         sns.heatmap(baseline_corr_mean[run], cmap='viridis', ax=axes[0])
    #         axes[0].set_title(f"Baseline Run {run} Correlation")
    #         corr = np.corrcoef(raw.get_data())
    #         sns.heatmap(corr, cmap='viridis', ax=axes[1])
    #         axes[1].set_title(f"Subject {subject} Run {run} Correlation")
    #         plt.tight_layout()
    #         plt.savefig(f"{out_dir}/corr_{run}.png")
    #     else:
    #         corr = np.corrcoef(raw.get_data())
    #         plt.figure()
    #         sns.heatmap(corr, cmap='viridis')
    #         plt.title(f"Subject {subject} Run {run} Correlation")
    #         plt.savefig(f"{out_dir}/corr_{run}.png")
    #     # bool_print correlation summary
    #     triu_idx = np.triu_indices_from(corr, k=1)
    #     subj_mean_corr = corr[triu_idx].mean()
    #     if baseline_n > 0 and 'baseline_corr_mean' in locals():
    #         base_corr = baseline_corr_mean[run]
    #         base_mean_corr = base_corr[triu_idx].mean()
    #         bool_print(f"Run {run} mean channel correlation: subject {subj_mean_corr:.2f}, baseline {base_mean_corr:.2f}")
    #     else:
    #         bool_print(f"Run {run} mean channel correlation: subject {subj_mean_corr:.2f}")
    #     # Preprocess
    #     data = notch_highpass_rereference(raw)
    #     data = auto_reject_channels(data); bads.append(len(data.info['bads']))
    #     if info_csp is None: info_csp = data.info
    #     data = run_ica(data, ['eye blink','heart beat','muscle artifact','other'])
    #     data = bandpass_filter(data, l_freq=1, h_freq=50)
    #     epochs = epoch_data(data, e_start=-1, e_end=4); before.append(len(epochs))
    #     clean = auto_reject_trials(epochs); after.append(len(clean)); clean = clean.apply_baseline((-1,0))
    #     X, y = clean.get_data(), clean.events[:,2]
    #     mapping = {lab:i for i,lab in enumerate(np.unique(y))}
    #     y = np.array([mapping[v] for v in y])
    #     Xs.append(X); ys.append(y)
    # # Plot summary bad channels
    # fig, ax = plt.subplots(figsize=(6,4))
    # ax.bar([str(r) for r in runs], bads, color='tab:blue', label=f'Sub {subject}')
    # ax.set_xlabel('Run'); ax.set_ylabel('Bad channels'); ax.set_title(f'Subject {subject} - Bad Channels per Run')
    # if baseline_n > 0 and 'baseline_bad_counts_by_run' in locals():
    #     # overlay baseline avg bads if computed
    #     mean_bads = [np.mean(baseline_bad_counts_by_run[run]) for run in runs]
    #     ax.plot([str(r) for r in runs], mean_bads, color='gray', linestyle='--', marker='o', label='Baseline avg')
    #     ax.legend()
    # plt.savefig(f"{out_dir}/bads.png")
    # # bool_print bad channel summary
    # bads_dict = {str(r): b for r,b in zip(runs, bads)}
    # bool_print(f"Bad channels per run: {bads_dict}")
    # if baseline_n > 0 and 'baseline_bad_counts_by_run' in locals():
    #     mean_bads = [np.mean(baseline_bad_counts_by_run[run]) for run in runs]
    #     mean_bads_dict = {str(r): mb for r,mb in zip(runs, mean_bads)}
    #     bool_print(f"Baseline average bad channels per run: {mean_bads_dict}")
    # # Plot epochs before/after
    # fig, ax = plt.subplots(figsize=(6,4))
    # pos = np.arange(len(runs)); width=0.35
    # ax.bar(pos - width/2, before, width, label='Before'); ax.bar(pos + width/2, after, width, label='After')
    # ax.set_xticks(pos); ax.set_xticklabels([str(r) for r in runs])
    # ax.set_xlabel('Run'); ax.set_ylabel('Epoch count'); ax.set_title('Epochs Before/After Rejection')
    # ax.legend()
    # plt.savefig(f"{out_dir}/epochs.png")
    # # bool_print epochs counts
    # epochs_counts = {str(r): {'before': before[i], 'after': after[i]} for i,r in enumerate(runs)}
    # bool_print(f"Epoch counts before/after rejection: {epochs_counts}")
    # CSP + LDA
    # X_all = np.concatenate(Xs,0); 
    # y_all = np.concatenate(ys,0)
    # csp = CSP(n_components=4, log=True)
    # Xc = csp.fit_transform(X_all, y_all)
    # cv = StratifiedKFold(5, shuffle=True, random_state=0)
    # scores = cross_val_score(LDA(), Xc, y_all, cv=cv)
    # bool_print(f"CSP+LDA CV: {scores.mean():.3f} ± {scores.std():.3f}")
    # if scores.mean() < 0.5:
    #     bool_print(f"Raw CSP+LDA below chance (0.5) by {0.5 - scores.mean():.3f}")
    # else:
    #     bool_print(f"Raw CSP+LDA above chance by {scores.mean() - 0.5:.3f}")
    # fig = csp.plot_patterns(info_csp, ch_type='eeg'); fig.savefig(f"{out_dir}/csp_patterns.png"); plt.close(fig)
    # Processed data analysis using feature_modules
    X_proc, y_proc, info_proc = load_processed_data(subject, runs, True)

    X_proc = torch.FloatTensor(X_proc).to(DEVICE)
    y_proc = torch.LongTensor(y_proc).to(DEVICE)

    # # CSP features on processed data via module
    # csp_mod = CSPModule(n_components=4, per_subject=False, filter_bank=True,
    #                     bands=[(8, 10), (10, 12), (12, 15), (15, 18), (18, 21), (21, 24), (24, 27), (27, 30)])
    # csp_mod.fit(X_proc, y_proc)
    # # CSP transform returns torch.Tensor, move to CPU and NumPy for sklearn
    # Xc_proc = csp_mod.transform(X_proc)

    # # Classification
    # scores_proc = cross_val_score(LDA(), Xc_proc.cpu(), y_proc.cpu(), cv=StratifiedKFold(5, shuffle=True, random_state=0))
    # bool_print(f"Processed CSP+LDA CV: {scores_proc.mean():.3f} ± {scores_proc.std():.3f}")
    # # Processed CSP+LDA fold accuracies
    # bool_print(f"Processed CSP+LDA fold accuracies: {scores_proc.tolist()}")
    # if scores_proc.mean() < 0.5:
    #     bool_print(f"Processed CSP+LDA below chance by {0.5 - scores_proc.mean():.3f}")
    # else:
    #     bool_print(f"Processed CSP+LDA above chance by {scores_proc.mean() - 0.5:.3f}")
    # # Plot processed CSP+LDA accuracies
    # plt.figure(); plt.bar(np.arange(1,len(scores_proc)+1), scores_proc); plt.xlabel('Fold'); plt.ylabel('Accuracy'); plt.title('Processed CSP+LDA CV'); plt.savefig(f"{out_dir}/processed_csp_lda_cv.png"); plt.close()
    # # Scatter first two CSP components
    # plt.figure(); sc = plt.scatter(Xc_proc[:,0], Xc_proc[:,1], c=y_proc, cmap='tab10', alpha=0.7); plt.xlabel('CSP Component 1'); plt.ylabel('CSP Component 2'); plt.title('Processed CSP Scatter'); legend = plt.legend(*sc.legend_elements(), title='Class'); plt.gca().add_artist(legend); plt.savefig(f"{out_dir}/processed_csp_scatter.png"); plt.close()
    # # Processed CSP scatter centroids
    # cent0 = Xc_proc[y_proc==0, :2].mean(axis=0)
    # cent1 = Xc_proc[y_proc==1, :2].mean(axis=0)
    # bool_print(f"Processed CSP scatter centroids: class0 ({cent0[0]:.3f}, {cent0[1]:.3f}), class1 ({cent1[0]:.3f}, {cent1[1]:.3f})")
    # # Class distribution
    # plt.figure(); counts = [np.sum(y_proc==lbl) for lbl in np.unique(y_proc)]; plt.bar(['Class 0','Class 1'], counts); plt.title('Processed Class Distribution'); plt.savefig(f"{out_dir}/processed_class_distribution.png"); plt.close()
    # # Processed class distribution counts
    # bool_print(f"Processed class distribution counts: {{'class0': {counts[0]}, 'class1': {counts[1]}}}")
    # # Average ERP across all channels per class
    # times = np.arange(X_proc.shape[2]) / info_proc['sfreq']
    # avg0 = X_proc[y_proc==0].mean(axis=(0,1))
    # avg1 = X_proc[y_proc==1].mean(axis=(0,1))
    # plt.figure(); plt.plot(times, avg0, label='Class 0'); plt.plot(times, avg1, label='Class 1'); plt.xlabel('Time (s)'); plt.ylabel('Amplitude (µV)'); plt.title('Processed Average ERP'); plt.legend(); plt.savefig(f"{out_dir}/processed_avg_erp.png"); plt.close()
    # # Processed average ERP peaks
    # peak0, t0 = avg0.max(), times[avg0.argmax()]
    # peak1, t1 = avg1.max(), times[avg1.argmax()]
    # bool_print(f"Processed Avg ERP peaks: class0 {peak0:.3f} µV at {t0:.3f}s; class1 {peak1:.3f} µV at {t1:.3f}s")
    # # Channel correlation heatmap
    # chan_data = X_proc.transpose(1,0,2).reshape(X_proc.shape[1], -1)
    # corr_proc = np.corrcoef(chan_data)
    # plt.figure(figsize=(8,6)); sns.heatmap(corr_proc, xticklabels=info_proc['ch_names'], yticklabels=info_proc['ch_names'], cmap='coolwarm', center=0); plt.title('Processed Channel Correlation'); plt.savefig(f"{out_dir}/processed_channel_correlation.png"); plt.close()
    # # Processed channel correlation stats
    # triu = np.triu_indices_from(corr_proc, k=1)
    # bool_print(f"Processed channel correlation: mean {corr_proc[triu].mean():.3f}, std {corr_proc[triu].std():.3f}")
    # =======================
    # Baseline feature comparison diagnostics
    # =======================
    if baseline_n > 0 and len(baseline_ids) > 0:
        bool_print("Baseline feature comparison diagnostics → subject vs. baseline group")
        # Load processed data for baseline subjects (reuse utility)
        X_base_proc, y_base_proc, subj_indices_base = load_multi_subject_data(baseline_ids, runs, True, max_subjects=baseline_n)

        X_base_proc = torch.FloatTensor(X_base_proc).to(DEVICE)
        y_base_proc = torch.LongTensor(y_base_proc).to(DEVICE)
        subj_indices_base = torch.IntTensor(subj_indices_base).to(DEVICE)

        # # --- CSP features (processed-data pipeline) ---
        # Xc_base = csp_mod.transform(X_base_proc)

        # subj_csp_mean = Xc_proc.mean(axis=0)
        # base_csp_mean = Xc_base.mean(axis=0)
        # base_csp_std  = Xc_base.std(axis=0) + 1e-12  # avoid zero
        # z_csp = (subj_csp_mean - base_csp_mean) / base_csp_std
        # # bool_print components where |z|>2
        # outlier_idx = torch.where(torch.abs(z_csp) > 2)[0]
        # bool_print(f"CSP component z-scores (subject mean vs. baseline):")
        # for i, z in enumerate(z_csp):
        #     flag = "*" if i in outlier_idx else ""
        #     bool_print(f"  Comp {i}: z={z:.2f}{flag}")
        # if len(outlier_idx) == 0:
        #     bool_print("  No CSP components deviate >2σ from baseline")
        # # --- Train baseline LDA on baseline subjects and test on subject ---
        # lda_feat = LDA()
        # scores_base = cross_val_score(lda_feat, Xc_base, y_base_proc, cv=StratifiedKFold(5, shuffle=True, random_state=0))
        # lda_feat.fit(Xc_base, y_base_proc)
        # acc_subj_on_base = lda_feat.score(Xc_proc, y_proc)
        # bool_print("\nBaseline-trained LDA:")
        # bool_print(f"  Baseline CV accuracy: {scores_base.mean():.3f} ± {scores_base.std():.3f}")
        # bool_print(f"  Subject accuracy when evaluated with baseline LDA: {acc_subj_on_base:.3f}")
        # if acc_subj_on_base < (scores_base.mean() - 2*scores_base.std()):
        #     bool_print("  → Subject classified much worse than baseline expectation (possible distribution shift)")
        # else:
        #     bool_print("  → Subject within baseline accuracy range")
        # --- Base model diagnostics with hybrid feature modules ---
        if base_model_path:
            bool_print("\nBase model classification diagnostics with hybrid feature modules:")
            clf, metadata = HybridModelClassifier.load_model(base_model_path)
            feature_module_configs = metadata.get('feature_modules')
            if feature_module_configs is None:
                raise ValueError("No feature module configuration found in model metadata.")
            # Build baseline modules once
            modules_base = []
            for cfg in feature_module_configs:
                name = cfg['name']
                params = cfg['params']
                if name == 'CSPModule':
                    mb = CSPModule(**params)
                    mb.fit(X_base_proc, y_base_proc, subject_indices=subj_indices_base)
                    modules_base.append(mb)
                elif name == 'SpectrogramModule':
                    sb = SpectrogramModule(**params)
                    modules_base.append(sb)
                else:
                    raise ValueError(f"Unsupported feature module type: {name}")
            # Dummy zero-output module for ablation
            class ZeroModule:
                def __init__(self, feature_dim):
                    self.feature_dim = feature_dim
                def fit(self, *args, **kwargs):
                    pass
                def transform(self, X, subject_indices=None):
                    n = X.shape[0]
                    device = X.device if torch.is_tensor(X) else DEVICE
                    return torch.zeros(n, self.feature_dim, device=device)
            # Baseline evaluation (once)
            clf.feature_modules = modules_base
            clf.use_feature_modules = True
            base_full = clf.evaluate(X_base_proc, y_base_proc, subject_indices=subj_indices_base)
            bool_print(f"Baseline full features accuracy: {base_full['accuracy']:.3f}")
            bool_print("Baseline cross-validated diagnostics:")
            kf = StratifiedKFold(5, shuffle=True, random_state=0)
            base_cv = [clf.evaluate(X_base_proc[idx], y_base_proc[idx], subject_indices=subj_indices_base[idx])['accuracy'] for _, idx in kf.split(X_base_proc.cpu(), y_base_proc.cpu())]
            bool_print(f"  Baseline CV: {np.mean(base_cv):.3f} ± {np.std(base_cv):.3f}")
            # Loop over each target subject
            for subj in subject_list:
                bool_print(f"\nDiagnostics for subject {subj}:")
                # Load processed data for this subject
                X_proc, y_proc, _ = load_processed_data(subj, runs, True)
                X_proc = torch.FloatTensor(X_proc).to(DEVICE)
                y_proc = torch.LongTensor(y_proc).to(DEVICE)
                # Build subject-specific modules
                modules_subj = []
                for cfg in feature_module_configs:
                    name = cfg['name']
                    params = cfg['params']
                    if name == 'CSPModule':
                        ms = CSPModule(**params)
                        zero_idx = torch.zeros(len(y_proc), dtype=int)
                        ms.fit(X_proc, y_proc, subject_indices=zero_idx)
                        modules_subj.append(ms)
                    elif name == 'SpectrogramModule':
                        ss = SpectrogramModule(**params)
                        modules_subj.append(ss)
                    else:
                        raise ValueError(f"Unsupported feature module type: {name}")
                # Subject full features evaluation
                clf.feature_modules = modules_subj
                clf.use_feature_modules = True
                subj_res = clf.evaluate(X_proc, y_proc, subject_indices=torch.zeros(len(y_proc), dtype=int))
                bool_print(f"  Subject full features accuracy: {subj_res['accuracy']:.3f}")
                # Subject cross-validated diagnostics
                bool_print("  Subject cross-validated diagnostics:")
                kf = StratifiedKFold(5, shuffle=True, random_state=0)
                subj_cv = [
                    clf.evaluate(X_proc[idx], y_proc[idx], subject_indices=torch.zeros(len(idx), dtype=int))['accuracy']
                    for _, idx in kf.split(X_proc.cpu(), y_proc.cpu())
                ]
                bool_print(f"    Subject CV: {np.mean(subj_cv):.3f} ± {np.std(subj_cv):.3f}")
                # Feature ablation diagnostics
                bool_print("  Feature ablation diagnostics:")
                clf.use_feature_modules = False
                res_raw = clf.evaluate(X_proc, y_proc, subject_indices=torch.zeros(len(y_proc), dtype=int))
                bool_print(f"    Raw only: {res_raw['accuracy']:.3f}")
                for cfg, real in zip(feature_module_configs, modules_subj):
                    name = cfg['name']
                    ablated = [ZeroModule(m.feature_dim) if m is real else m for m in modules_subj]
                    clf.feature_modules = ablated
                    clf.use_feature_modules = True
                    res_ab = clf.evaluate(X_proc, y_proc, subject_indices=torch.zeros(len(y_proc), dtype=int))
                    bool_print(f"    Raw + {name}: {res_ab['accuracy']:.3f}")
                # Restore full feature set
                clf.feature_modules = modules_subj
                clf.use_feature_modules = True


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--subject', default='022', help='Primary subject for standard analysis')
    p.add_argument('--subjects', nargs='+', default=None, help='List of subjects for base-model diagnostics')
    p.add_argument('--runs', nargs='+', type=int, default=[3,7,11])
    p.add_argument('--out_dir', default='/home/connor/LIINC/test_figures')
    p.add_argument('--baseline-n', type=int, default=20, help='Number of baseline subjects to compare')
    p.add_argument('--base-model-path', default=None, help='Path to pretrained Hybrid base model (.pt) for classification diagnostics')
    args = p.parse_args()
    mne.set_log_level('WARNING')
    main(args.subject, args.subjects, args.runs, os.path.join(args.out_dir, f'sub_{args.subject}'), args.baseline_n, args.base_model_path) 