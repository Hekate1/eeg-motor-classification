"""
Additional Model Architectures for Single Subject EEG Classification

This module provides additional EEG classification models:
1. A 1D Transformer optimized for EEG
2. A simple but effective 1D CNN model
3. A combined CNN-Transformer architecture

To be used with single_subject_classification.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EEG1DTransformer(nn.Module):
    """1D Transformer model optimized for EEG classification.
    
    Uses a time-first approach to embedding, processing the time dimension
    with self-attention, which is well-suited for capturing long-range temporal
    dependencies in EEG signals.
    """
    
    def __init__(self, n_channels, n_times, n_classes, 
                 d_model=64, nhead=4, num_layers=2, 
                 dropout=0.3, activation='gelu'):
        super(EEG1DTransformer, self).__init__()
        
        # Initial channel-wise convolution to extract local features
        self.local_feature_extractor = nn.Sequential(
            nn.Conv1d(n_channels, d_model, kernel_size=15, padding='same'),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Temporal embedding
        # We swap the channel and time dimensions and process time with self-attention
        self.pos_encoder = nn.Parameter(torch.zeros(1, n_times, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True  # Pre-LN Transformer (more stable training)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Global pooling (attention-based) across time dimension
        self.time_attention = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes)
        )
    
    def forward(self, x):
        """Forward pass through the model.
        
        Parameters:
        -----------
        x : tensor, shape (batch_size, n_channels, n_times)
            Input EEG data
        
        Returns:
        --------
        tensor, shape (batch_size, n_classes)
            Output class logits
        """
        batch_size, n_channels, n_times = x.shape
        
        # Extract local features along time dimension
        x = self.local_feature_extractor(x)  # (batch_size, d_model, n_times)
        
        # Reshape for transformer: (batch_size, n_times, d_model)
        x = x.transpose(1, 2)
        
        # Add positional encoding
        x = x + self.pos_encoder
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)  # (batch_size, n_times, d_model)
        
        # Apply attention-based global pooling across time
        attn_weights = self.time_attention(x)  # (batch_size, n_times, 1)
        x = torch.sum(x * attn_weights, dim=1)  # (batch_size, d_model)
        
        # Classification
        x = self.classifier(x)
        
        return x


class Compact1DCNN(nn.Module):
    """Compact 1D CNN model for EEG classification.
    
    A simple but effective architecture with 1D convolutions
    that process each channel's temporal patterns independently 
    before combining information across channels.
    """
    
    def __init__(self, n_channels, n_times, n_classes, dropout_rate=0.4):
        super(Compact1DCNN, self).__init__()
        
        # Channel-wise temporal convolutions (wide)
        self.temporal_block1 = nn.Sequential(
            nn.Conv1d(n_channels, n_channels*2, kernel_size=64, padding='same', groups=n_channels),
            nn.BatchNorm1d(n_channels*2),
            nn.ELU(),
            nn.AvgPool1d(kernel_size=4),
            nn.Dropout(dropout_rate)
        )
        
        # Channel-wise temporal convolutions (narrow)
        self.temporal_block2 = nn.Sequential(
            nn.Conv1d(n_channels*2, n_channels*4, kernel_size=16, padding='same', groups=n_channels*2),
            nn.BatchNorm1d(n_channels*4),
            nn.ELU(),
            nn.AvgPool1d(kernel_size=4),
            nn.Dropout(dropout_rate)
        )
        
        # Cross-channel integration
        self.channel_mixer = nn.Sequential(
            nn.Conv1d(n_channels*4, 64, kernel_size=8, padding='same'),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Dropout(dropout_rate)
        )
        
        # Calculate output size after pooling
        self.feature_size = 64 * (n_times // (4 * 4 * 2))
        
        # Global average pooling fallback if feature size is too small
        self.use_gap = (self.feature_size <= 64)
        if self.use_gap:
            self.feature_size = 64
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 32),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, n_classes)
        )
    
    def forward(self, x):
        """Forward pass through the model.
        
        Parameters:
        -----------
        x : tensor, shape (batch_size, n_channels, n_times)
            Input EEG data
        
        Returns:
        --------
        tensor, shape (batch_size, n_classes)
            Output class logits
        """
        # Apply temporal blocks
        x = self.temporal_block1(x)
        x = self.temporal_block2(x)
        
        # Apply channel mixer
        x = self.channel_mixer(x)
        
        # Global average pooling if needed, otherwise flatten
        if self.use_gap:
            x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        else:
            x = x.reshape(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        
        return x


class CNN_Transformer_Hybrid(nn.Module):
    """Hybrid CNN-Transformer model for EEG classification.
    
    Combines the strengths of CNNs for local feature extraction
    with transformers for capturing global dependencies.
    """
    
    def __init__(self, n_channels, n_times, n_classes, 
                 d_model=64, nhead=4, num_layers=2, dropout=0.4):
        super(CNN_Transformer_Hybrid, self).__init__()
        
        # CNN feature extraction path
        self.cnn_features = nn.Sequential(
            # Temporal convolution
            nn.Conv2d(1, 16, (1, 32), padding='same'),
            nn.BatchNorm2d(16),
            nn.GELU(),
            
            # Spatial convolution
            nn.Conv2d(16, 32, (n_channels, 1)),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d((1, 4)),
            nn.Dropout(dropout)
        )
        
        # Calculate feature dimensions after CNN
        self.cnn_output_size = 32 * (n_times // 4)
        
        # Embedding layer to project CNN features to transformer dimension
        self.cnn_to_transformer = nn.Linear(32, d_model)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, n_times // 4, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Classification head with shortcut connection
        self.classifier = nn.Sequential(
            nn.Linear(d_model + 32, 64),  # Concatenate transformer output with CNN features
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )
    
    def forward(self, x):
        """Forward pass through the model.
        
        Parameters:
        -----------
        x : tensor, shape (batch_size, n_channels, n_times)
            Input EEG data
        
        Returns:
        --------
        tensor, shape (batch_size, n_classes)
            Output class logits
        """
        batch_size = x.size(0)
        
        # Add channel dimension for 2D convolution
        x_cnn = x.unsqueeze(1)  # (batch_size, 1, n_channels, n_times)
        
        # Apply CNN feature extraction
        x_cnn = self.cnn_features(x_cnn)  # (batch_size, 32, 1, n_times/4)
        x_cnn = x_cnn.squeeze(2)  # (batch_size, 32, n_times/4)
        
        # Store CNN features for skip connection
        cnn_features_for_skip = x_cnn.mean(dim=2)  # (batch_size, 32)
        
        # Prepare for transformer
        x_trans = x_cnn.transpose(1, 2)  # (batch_size, n_times/4, 32)
        x_trans = self.cnn_to_transformer(x_trans)  # (batch_size, n_times/4, d_model)
        
        # Add positional encoding
        x_trans = x_trans + self.pos_encoder
        
        # Apply transformer encoder
        x_trans = self.transformer_encoder(x_trans)  # (batch_size, n_times/4, d_model)
        
        # Global average pooling over time dimension
        x_trans = x_trans.mean(dim=1)  # (batch_size, d_model)
        
        # Combine transformer output with CNN features
        x_combined = torch.cat([x_trans, cnn_features_for_skip], dim=1)
        
        # Classification
        x = self.classifier(x_combined)
        
        return x


def create_transformer_classifier(n_channels, n_times, n_classes, device):
    """Create a 1D Transformer classifier for EEG data
    
    Parameters:
    -----------
    n_channels : int
        Number of EEG channels
    n_times : int
        Number of time points
    n_classes : int
        Number of output classes
    device : torch.device
        Device to use for training
        
    Returns:
    --------
    torch.nn.Module
        Initialized transformer model
    """
    model = EEG1DTransformer(
        n_channels=n_channels,
        n_times=n_times,
        n_classes=n_classes,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.3
    ).to(device)
    
    return model


def create_cnn_classifier(n_channels, n_times, n_classes, device):
    """Create a 1D CNN classifier for EEG data
    
    Parameters:
    -----------
    n_channels : int
        Number of EEG channels
    n_times : int
        Number of time points
    n_classes : int
        Number of output classes
    device : torch.device
        Device to use for training
        
    Returns:
    --------
    torch.nn.Module
        Initialized CNN model
    """
    model = Compact1DCNN(
        n_channels=n_channels,
        n_times=n_times,
        n_classes=n_classes,
        dropout_rate=0.4
    ).to(device)
    
    return model


def create_hybrid_classifier(n_channels, n_times, n_classes, device):
    """Create a hybrid CNN-Transformer classifier for EEG data
    
    Parameters:
    -----------
    n_channels : int
        Number of EEG channels
    n_times : int
        Number of time points
    n_classes : int
        Number of output classes
    device : torch.device
        Device to use for training
        
    Returns:
    --------
    torch.nn.Module
        Initialized hybrid model
    """
    model = CNN_Transformer_Hybrid(
        n_channels=n_channels,
        n_times=n_times,
        n_classes=n_classes,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.4
    ).to(device)
    
    return model