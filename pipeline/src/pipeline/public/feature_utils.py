import numpy as np
from scipy.signal import welch
from mne.decoding import CSP
from mne.filter import filter_data
import torch
import hashlib
DEBUG = False                # Global debug flag to control printouts

# Debug print function
def dprint(*args, **kwargs):
    """Only print if DEBUG is True"""
    if DEBUG:
        print(*args, **kwargs)

# Cache for CSPTransformer instances to avoid re-fitting
_csp_transformer_cache = {}
# Cache for pre-filtered band data in filter-bank CSP
_fb_filter_data_cache = {}

class CSPTransformer:
    def __init__(self, X, y, per_subject=True, subject_indices=None,
                 n_components=4, filter_bank=False,
                 bands=[(8,12),(13,30)], sfreq=160):
        self.filter_bank = filter_bank
        self.bands       = bands
        self.sfreq       = sfreq
        self.n_components= n_components
        self.per_subject = per_subject

        if per_subject:
            assert subject_indices is not None, "subject_indices must be provided if per_subject is True"
            self.subject_csps = {}
            if not self.filter_bank:
                # per-subject CSP without filter bank
                self.transform = self.per_subject_transform
                for subj in np.unique(subject_indices):
                    mask = subject_indices == subj
                    Xs, ys = X[mask], y[mask]
                    c = CSP(self.n_components, log=True).fit(Xs, ys)
                    self.subject_csps[subj] = c
            else:
                # per-subject CSP with filter bank (pre-filter once per band)
                self.transform = self.filter_bank_per_subject_transform
                # Pre-filter all data for each band once (raw data cache)
                band_key = (id(X), tuple(self.bands), self.sfreq)
                if band_key in _fb_filter_data_cache:
                    X_bands = _fb_filter_data_cache[band_key]
                    dprint(f"Using cached band-filtered data for key {band_key}")
                else:
                    X_bands = [filter_data(X, self.sfreq, low, high, verbose=False, n_jobs=-1)
                               for (low, high) in self.bands]
                    _fb_filter_data_cache[band_key] = X_bands
                for subj in np.unique(subject_indices):
                    mask = subject_indices == subj
                    # Fit one CSP per band on masked, pre-filtered data
                    csps = [CSP(self.n_components, log=True).fit(X_bands[i][mask], y[mask])
                           for i in range(len(self.bands))]
                    self.subject_csps[subj] = csps
        else:
            self.transform = self.global_transform
            if not self.filter_bank:
                self.global_csp = CSP(n_components, log=True).fit(X, y)
            else:
                self.global_csps = []
                for band in self.bands:
                    Xb = filter_data(X, self.sfreq, band[0], band[1], verbose=False, n_jobs=-1)
                    self.global_csps.append(CSP(n_components, log=True).fit(Xb, y))

    def global_transform(self, X, subject_indices=None):
        assert not self.per_subject
        if not self.filter_bank:
            return self.global_csp.transform(X)
        else:
            n_trials = len(X)
            nb      = len(self.global_csps)
            out     = np.zeros((n_trials, nb*self.n_components))
            for i, csp in enumerate(self.global_csps):
                Xb = filter_data(X, self.sfreq,
                                 self.bands[i][0], self.bands[i][1],
                                 verbose=False, n_jobs=-1)
                out[:, i*self.n_components:(i+1)*self.n_components] = csp.transform(Xb)
            return out
        
    def per_subject_transform(self, X, subject_indices):
        assert (not self.filter_bank) and self.per_subject and (subject_indices is not None)

        # Pre-allocate for per-subject CSP features
        out = np.zeros((X.shape[0], self.n_components), dtype=X.dtype)

        # Apply CSP transform per subject
        for sid, csp in self.subject_csps.items():
            idx = np.where(subject_indices == sid)[0]
            if idx.size:
                out[idx] = csp.transform(X[idx])

        return out

    def filter_bank_per_subject_transform(self, X, subject_indices):
        assert self.filter_bank and self.per_subject and subject_indices is not None

        # Pre-filter input X for each band once to speed up per-subject transform
        # Compute a hash of X's raw bytes to key the cache (handles different array objects with same data)
        X_arr = np.ascontiguousarray(X)
        hash_digest = hashlib.sha1(X_arr.view(np.uint8)).hexdigest()
        band_key = (hash_digest, tuple(self.bands), self.sfreq)
        if band_key in _fb_filter_data_cache:
            X_bands = _fb_filter_data_cache[band_key]
            dprint(f"Using cached band-filtered data for key {band_key}")
        else:
            X_bands = [filter_data(X_arr, self.sfreq, low, high, verbose=False, n_jobs=-1)
                       for (low, high) in self.bands]
            _fb_filter_data_cache[band_key] = X_bands
        # Per-subject filter-bank transform
        nb = len(self.bands)
        out = np.zeros((len(X), nb * self.n_components))
        for subj, csps in self.subject_csps.items():
            idx = np.where(subject_indices == subj)[0]
            if not idx.size:
                continue
            # Stack CSP outputs per band for this subject
            feats = [csp.transform(X_bands[i][idx]) for i, csp in enumerate(csps)]
            out[idx] = np.hstack(feats)
        return out
