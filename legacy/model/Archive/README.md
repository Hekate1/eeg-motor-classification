# Hybrid CNN/Transformer for EEG Classification

This repository contains a hybrid CNN/Transformer architecture for EEG classification, particularly focused on motor imagery tasks.

## Project Structure

```
├── models/
│   ├── __init__.py                  # Package initialization
│   ├── hybrid_cnn_transformer.py    # Core model architecture
│   ├── data_augmenter.py            # Data augmentation classes
│   ├── model_classifier.py          # Model training and evaluation wrapper
│   └── evaluation.py                # Evaluation functions
├── train_evaluate_model.py          # Main script for training/evaluation
├── hybrid_cnn_transformer.py        # Original combined file (legacy)
└── README.md                        # This file
```

## Features

- **Hybrid Architecture**: Combines CNN for spatial feature extraction with Transformer for temporal dependencies
- **Multi-modal Fusion**: Integrates raw EEG with Common Spatial Pattern (CSP) and frequency-domain features
- **Robust Data Augmentation**: Implements physiologically plausible augmentation techniques for EEG
- **Transfer Learning**: Supports pretraining on multiple subjects and fine-tuning for subject-specific models
- **Balanced Prediction**: Helps prevent the model from getting stuck predicting only the majority class

## Usage

### Cross-validation on a Single Subject

```bash
python train_evaluate_model.py cv --subject 1 --data_dir /path/to/data --output_dir results/cv
```

### Pretrain on Multiple Subjects

```bash
python train_evaluate_model.py pretrain --subjects 1 2 3 4 --data_dir /path/to/data --output_dir results/pretrain
```

### Fine-tune for a Specific Subject

```bash
python train_evaluate_model.py finetune --subject 9 --model results/pretrain/pretrained_model.pt --data_dir /path/to/data --output_dir results/finetune
```

### Evaluate Model on Multiple Subjects

```bash
python train_evaluate_model.py evaluate --subjects 1 2 3 --model results/pretrain/pretrained_model.pt --data_dir /path/to/data --output_dir results/evaluate
```

## Data Format

The model expects EEG data in the following format:
- Raw EEG: [n_trials, n_channels, n_times]
- Labels: [n_trials] (integer class labels)

## Key Enhancements

1. **Balanced Prediction**: Added a `force_balanced` parameter that helps break out of the "predict one class" trap through logit adjustment
2. **Optimized Feature Integration**: Better fusion of CSP and frequency features with raw EEG signals
3. **Improved Augmentation**: Enhanced data augmentation techniques that are physiologically plausible for EEG
4. **Robust Fine-tuning**: Specialized fine-tuning process with adaptive learning rates and stronger class balancing

## Performance Considerations

- The model performs best when both CSP and frequency features are available
- For subject-specific fine-tuning, more aggressive class balancing is applied to prevent overfitting to the majority class
- The validation process evaluates both with and without balanced prediction, allowing you to see both the raw model behavior and the corrected behavior
