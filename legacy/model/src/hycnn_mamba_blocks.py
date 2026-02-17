import torch
import torch.nn as nn
from typing import Optional

# Reuse PositionalEncoding and EEGCNNEncoder from hycnn_blocks to avoid duplicate code
from hycnn_blocks import PositionalEncoding, EEGCNNEncoder, DynamicGatedFusion

__all__ = [
    'MambaBlock',
    'HybridCNNMamba'
]

class MambaBlock(nn.Module):
    """A lightweight approximation of the Mamba selective state-space model.
    
    NOTE
    ----
    This is **not** the full Mamba implementation from the official paper, but a
    simplified gated depth-wise convolutional block that empirically works well
    on 1-D biosignals while keeping the dependency footprint minimal.  If you
    wish to swap this for the official implementation just replace the forward
    pass and remove the convolutional approximation.
    """

    def __init__(self, d_model: int, d_ff: Optional[int] = None,
                 kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4

        # Depth-wise gated convolution – acts as SSM kernel proxy
        self.dwconv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        # Point-wise convolution for mixing channels
        self.pwconv = nn.Conv1d(d_model, d_model, kernel_size=1)

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        # Layer norms and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Gating function
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, C]
        residual = x

        # Convolutional SSM projection
        x_conv = x.transpose(1, 2)  # [B, C, T]
        x_conv = self.dwconv(x_conv)
        x_conv = self.pwconv(x_conv)
        x_conv = x_conv.transpose(1, 2)  # [B, T, C]

        # Gating (element-wise)
        x = self.gate(x_conv) * x_conv
        x = self.norm1(residual + self.dropout(x))

        # Feed-forward
        residual = x
        x = self.ff(x)
        x = self.norm2(residual + self.dropout(x))
        return x


class HybridCNNMamba(nn.Module):
    """Hybrid CNN + Mamba SSM model for EEG classification.

    The interface matches ``HybridCNNTransformer`` so that the surrounding
    training code can remain unchanged.  Internally, the transformer encoder
    stack is replaced by a stack of ``MambaBlock`` modules.
    """

    def __init__(self, n_channels, n_times, n_classes,
                 feature_dim=128,
                 embedding_dim=128, n_heads=4, n_layers=2,
                 dropout=0.2, use_feature_modules=True,
                 seq_len_override=None):
        super().__init__()

        # Store parameters for serialization
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.use_feature_modules = use_feature_modules

        # CNN front-end
        self.cnn_encoder = EEGCNNEncoder(
            n_channels=n_channels,
            n_times=n_times,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

        # Output temporal resolution after CNN
        self.n_times_out = n_times // 32

        # Feature encoding branch (e.g., CSP or other handcrafted features)
        if use_feature_modules:
            self.feature_encoder = nn.Sequential(
                nn.Linear(feature_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.feature_encoder = None

        # Sequence length after CNN
        self.seq_len = seq_len_override if seq_len_override is not None else self.n_times_out

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=embedding_dim, max_len=self.seq_len, dropout=dropout
        )

        # Mamba encoder layers
        self.transformer_layers = nn.ModuleList([
            MambaBlock(d_model=embedding_dim, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Classification heads (updated to include dynamic gated fusion)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.fusion = DynamicGatedFusion(embedding_dim, embedding_dim, mode="concat")
        init_class_dim = self.fusion.out_dim

        self.classifier_body = nn.Sequential(
            nn.Linear(init_class_dim, init_class_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.adapter_hidden_dim = init_class_dim // 4
        self.adapter_head = nn.Sequential(
            nn.Linear(init_class_dim, self.adapter_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.combiner_hidden_dim = init_class_dim // 4
        self.combiner_head = nn.Sequential(
            nn.Linear((init_class_dim // 2) + self.adapter_hidden_dim, self.combiner_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.combiner_hidden_dim, n_classes),
        )

        self._init_weights()

    # Weight initialisation identical to original
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    # Forward pass mirrors original model
    def forward(self, x_raw, x_feat=None, attention_mask=None):
        batch_size = x_raw.size(0)

        # CNN features
        cnn_features = self.cnn_encoder(x_raw)  # [B, T', C]

        # Use raw CNN features as sequence input
        sequence = cnn_features  # [B, seq_len, C]
        sequence = self.pos_encoder(sequence)

        # Build full attention mask (Mamba implementation ignores mask but keep for parity)
        if attention_mask is not None:
            seq_len = sequence.size(1)
            if attention_mask.size(1) > seq_len:
                attention_mask = attention_mask[:, :seq_len]
            elif attention_mask.size(1) < seq_len:
                pad = torch.ones(batch_size, seq_len - attention_mask.size(1), device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, pad], dim=1)
            mask_expanded = attention_mask.unsqueeze(-1)
        else:
            mask_expanded = None

        # Apply Mamba layers
        for layer in self.transformer_layers:
            sequence = layer(sequence)
            if mask_expanded is not None:
                sequence = sequence * mask_expanded

        # Pooling – mean over sequence length
        if mask_expanded is not None:
            pooled = sequence.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = torch.mean(sequence, dim=1)

        # Forward classifier pipeline with dynamic gated fusion
        raw_feat = self.layer_norm(pooled)
        # Encode external features if provided
        feat = self.feature_encoder(x_feat) if (self.feature_encoder is not None and x_feat is not None) else None
        fused, _ = self.fusion(raw_feat, feat)
        hidden1 = self.classifier_body(fused)
        hidden2 = self.adapter_head(fused)
        combined_hidden = torch.cat([hidden1, hidden2], dim=1)
        logits = self.combiner_head(combined_hidden)
        return logits 