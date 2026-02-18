"""
Models package for EEG classification.

This package contains modules for model architecture, training, and evaluation.
"""

from models.hybrid_cnn_transformer import (
    HybridCNNTransformer,
    extract_csp_features,
    extract_frequency_features
)
from models.model_classifier import (
    HybridModelClassifier,
    subject_specific_normalize
)
from models.data_augmenter import (
    EEGDataAugmenter,
    OptimizedAugmenter,
    BasicAugmenter
)

__all__ = [
    'HybridCNNTransformer',
    'extract_csp_features',
    'extract_frequency_features',
    'HybridModelClassifier',
    'subject_specific_normalize',
    'EEGDataAugmenter',
    'OptimizedAugmenter',
    'BasicAugmenter'
] 