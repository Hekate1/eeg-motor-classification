import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    """
    Lightweight multi-head self-attention module
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out = nn.Linear(d_model, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # Linear projections and reshape
        q = self.query(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.key(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.value(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            # Determine large negative fill value based on dtype to avoid float16 overflow
            fill_value = -torch.finfo(scores.dtype).max
            # Handle different mask formats
            if mask.dim() == 2:
                # Convert 1D mask [batch_size, seq_len] to attention mask [batch_size, 1, 1, seq_len]
                mask = mask.unsqueeze(1).unsqueeze(2)
                scores = scores.masked_fill(mask == 0, fill_value)
            elif mask.dim() == 3:
                # Handle 2D mask [batch_size, seq_len, seq_len]
                mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
                scores = scores.masked_fill(mask == 0, fill_value)
        
        # Apply softmax and dropout
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)
        
        # Reshape and concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.out(context)
        
        return output


class TransformerEncoderLayer(nn.Module):
    """
    Lightweight transformer encoder layer
    """
    def __init__(self, d_model, n_heads, d_ff=1024, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        # Multi-head self-attention
        self.self_attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution for EEG data
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DepthwiseSeparableConv, self).__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            padding='same'
        )
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1
        )
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class SpecCNN(nn.Module):
    """
    2-D CNN for log-spectrograms shaped [B, C, F, T].
    First conv is **depth-wise over channels**, second mixes channels.
    """
    def __init__(self, in_ch, hidden=64, out_dim=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            # depth-wise: learns per-channel freq/time filters
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch),
            nn.BatchNorm2d(in_ch), nn.GELU(),

            # point-wise: mixes channels
            nn.Conv2d(in_ch, hidden, kernel_size=1),
            nn.BatchNorm2d(hidden), nn.GELU(),

            # one more conv block
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden),
            nn.BatchNorm2d(hidden), nn.GELU(),

            nn.Conv2d(hidden, 2*hidden, kernel_size=1),
            nn.BatchNorm2d(2*hidden), nn.GELU(),

            nn.AdaptiveAvgPool2d(1),   # [B, 2h, 1, 1]
            nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(2*hidden, out_dim)
        )

    def forward(self, x):             # x: [B, C, F, T]
        return self.net(x)

class EEGCNNEncoder(nn.Module):
    """
    CNN encoder for EEG data inspired by EEGNet
    """
    def __init__(self, n_channels, n_times, embedding_dim=128, dropout=0.2):
        super(EEGCNNEncoder, self).__init__()
        
        self.n_times = n_times
        
        # First block: Temporal convolution
        self.block1 = nn.Sequential(
            # Initial temporal convolution
            nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(16),
            # Depthwise convolution
            nn.Conv2d(16, 32, (n_channels, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout)
        )

        # Original implementation for new models
        self.block2 = nn.Sequential(
            # Separable convolution
            nn.Conv2d(32, 32, (1, 16), padding=(0, 8), groups=32, bias=False),
            nn.Conv2d(32, 32, (1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout)
        )
        
        # Calculate output size
        self.n_times_out = n_times // 32
        
        # Final convolution to reduce to embedding dimension
        self.final_conv = nn.Conv1d(32, embedding_dim, 1)
        self.flat_size = embedding_dim * self.n_times_out
        
    def forward(self, x):
        """Forward pass for the CNN encoder"""
        # Input shape: [batch_size, n_channels, n_times]
        batch_size = x.size(0)
        
        # Debug: Check for extreme values or all zeros in input data
        if hasattr(torch, 'allclose'):
            if torch.allclose(x, torch.zeros_like(x), atol=1e-8):
                print(f"WARNING: Input to CNN encoder contains all zeros or near zeros!")
                print(f"Input min: {x.min().item():.10f}, max: {x.max().item():.10f}, mean: {x.mean().item():.10f}")
                print(f"Input shape: {x.shape}")
                # Use a small random signal to avoid complete zeroing
                x = x + torch.randn_like(x) * 1e-4
        
        # Check for NaN or infinity
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("WARNING: Input contains NaN or infinity values!")
            # Replace NaN/Inf with small random values
            x = torch.where(torch.isnan(x) | torch.isinf(x), torch.randn_like(x) * 1e-4, x)
        
        
        
        # Reshape for 2D convolution
        x = x.unsqueeze(1)  # [batch_size, 1, n_channels, n_times]
        
        # Apply first block: Temporal and spatial convolutions
        x = self.block1(x)  # [batch_size, 32, 1, n_times/4]
        
        # Apply second block: Separable convolution
        x = self.block2(x)  # [batch_size, 32, 1, n_times/32]
        
        # Reshape for final convolution
        x = x.squeeze(2)  # [batch_size, 32, n_times/32]
        
        # Apply final convolution
        x = self.final_conv(x)  # [batch_size, embedding_dim, n_times/32]
        
        # Permute to sequence format
        x = x.permute(0, 2, 1)  # [batch_size, n_times/32, embedding_dim]
        
        return x


class GatedFusion(nn.Module):
    """
    Generic 2-branch soft-gated fusion.
    """
    def __init__(self, branch_dims, mode="concat"):
        super().__init__()
        # Store expected branch dimensions for handling missing branches
        self.branch_dims = branch_dims
        self.mode = mode
        self.alpha = nn.Parameter(torch.zeros(len(branch_dims)))
        if mode == "concat":
            self.post = nn.Linear(sum(branch_dims), 256)
        else:
            self.post = nn.Linear(branch_dims[0], 256)
        # Expose the output feature dimension of the fusion
        self.out_dim = self.post.out_features

    def forward(self, *branches):
        g = torch.sigmoid(self.alpha)
        # Prepare gated tensors, replacing missing branches with zeros
        tensors = []
        # Use first non-None branch as reference for batch size, device, dtype
        ref = next((b for b in branches if b is not None), None)
        for i, dim in enumerate(self.branch_dims):
            # Get branch tensor or None if missing
            b = branches[i] if i < len(branches) else None
            if b is None:
                # Create zero tensor for missing branch
                batch_size = ref.shape[0]
                device = ref.device
                dtype = ref.dtype
                b = torch.zeros(batch_size, dim, device=device, dtype=dtype)
            # Apply gating scalar
            tensors.append(g[i] * b)
        if self.mode == "concat":
            fused = torch.cat(tensors, dim=1)
        else:
            fused = torch.stack(tensors).sum(0)
        return nn.functional.relu(self.post(fused))
    
class DynamicGatedFusion(nn.Module):
    def __init__(self, d_raw, d_feat, hidden=64, mode="concat"):
        super().__init__()
        self.d_raw = d_raw
        self.d_feat = d_feat
        self.mode = mode
        # Fusion gate based on raw features, feature embeddings, and reliability scalar
        self.fc_alpha = nn.Sequential(
            nn.Linear(d_raw + d_feat + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1)        # single gating variable
        )
        self.post = nn.Linear(d_raw + d_feat if mode=="concat" else d_raw, 256)
        self.out_dim = self.post.out_features

    def forward(self, raw, feat, rel=None):
        # 1. produce per-sample gates
        assert raw is not None or feat is not None
        if raw is None:
            raw = torch.zeros(feat.shape[0], self.d_raw, device=feat.device, dtype=feat.dtype)
        if feat is None:
            feat = torch.zeros(raw.shape[0], self.d_feat, device=raw.device, dtype=raw.dtype)
        # If reliability not provided, default to zero
        if rel is None:
            rel = torch.zeros(raw.size(0), 1, device=raw.device, dtype=raw.dtype)
        # Compute single gating weight for feature branch
        beta = torch.sigmoid(self.fc_alpha(torch.cat([raw, feat, rel], dim=1)))  # [B,1]
        g_feat = beta
        g_raw = 1.0 - beta

        # 2. apply gates
        if self.mode == "concat":
            fused = torch.cat([g_raw*raw, g_feat*feat], dim=1)
        else:  # additive
            fused = g_raw*raw + g_feat*feat

        return nn.functional.relu(self.post(fused)), beta.detach()   # return β for logging

class HybridCNNTransformer(nn.Module):
    """
    Hybrid CNN/Transformer model for EEG classification
    with CSP integration
    """
    def __init__(self, n_channels, n_times, n_classes, 
                 embedding_dim=128, feature_dim=128,
                 n_heads=4, n_layers=2, dropout=0.2, 
                 use_feature_modules=True, seq_len_override=None):
        super(HybridCNNTransformer, self).__init__()
        
        # Store input dimensions as attributes for model serialization
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.use_feature_modules = use_feature_modules
        self.feature_dim = feature_dim
        
        # LDA reliability weights and bias for CSP features
        self.register_buffer('lda_W', torch.zeros(1, feature_dim))
        self.register_buffer('lda_b', torch.zeros(1))
        
        # CNN encoder for raw EEG
        self.cnn_encoder = EEGCNNEncoder(
            n_channels=n_channels,
            n_times=n_times,
            embedding_dim=embedding_dim,
            dropout=dropout
        )
        
        # Feature size calculation
        self.n_times_out = n_times // 32
        
        # Feature encoding branch (e.g., CSP or other handcrafted features)
        if use_feature_modules:
            self.feature_encoder = nn.Sequential(
                nn.Linear(feature_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        else:
            self.feature_encoder = None
        
        # Calculate total sequence length for transformer
        self.seq_len = seq_len_override if seq_len_override is not None else self.n_times_out

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=embedding_dim,
            max_len=self.seq_len,
            dropout=dropout
        )
        
        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=embedding_dim,
                n_heads=n_heads,
                d_ff=embedding_dim * 4,
                dropout=dropout
            ) for _ in range(n_layers)
        ])
        
        # Layer norm before classification
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # Gated fusion between raw and CSP features
        self.fusion = DynamicGatedFusion(embedding_dim, embedding_dim, mode="concat")
        init_class_dim = self.fusion.out_dim
        
        # Network 1 (primary classifier): larger frozen network
        self.classifier_body = nn.Sequential(
            nn.Linear(init_class_dim, init_class_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Network 2 (adapter): smaller network in parallel
        self.adapter_hidden_dim = init_class_dim // 4
        self.adapter_head = nn.Sequential(
            nn.Linear(init_class_dim, self.adapter_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Network 3 (combiner): combines hidden outputs of Networks 1 & 2
        self.combiner_hidden_dim = init_class_dim // 4
        self.combiner_head = nn.Sequential(
            nn.Linear((init_class_dim // 2) + self.adapter_hidden_dim, self.combiner_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.combiner_hidden_dim, n_classes)
        )

        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
    def forward(self, x_raw, x_feat=None, attention_mask=None):
        """Forward pass for the hybrid model with raw and feature streams"""
        # Get batch size
        batch_size = x_raw.size(0)
        
        # Process raw EEG with CNN
        cnn_features = self.cnn_encoder(x_raw)  # [batch_size, n_times_out, embedding_dim]
        if torch.isnan(cnn_features).any():
            print("NaN detected in CNN features")
        
        # Store CNN output shape if not already stored
        if not hasattr(self, 'cnn_output_shape'):
            self.cnn_output_shape = tuple(cnn_features.shape)
        
        # Use raw CNN features as transformer input
        sequence = cnn_features  # [batch_size, seq_len, embedding_dim]
        sequence = self.pos_encoder(sequence)
        if torch.isnan(sequence).any():
            print("NaN after positional encoding")
        
        # Handle variable-length sequences if attention mask is provided
        if attention_mask is not None:
            # Adjust mask for sequence length after features concatenation
            seq_len = sequence.size(1)
            if attention_mask.size(1) > seq_len:
                attention_mask = attention_mask[:, :seq_len]
            elif attention_mask.size(1) < seq_len:
                # Pad with ones (attend to all new tokens)
                padding = torch.ones(batch_size, seq_len - attention_mask.size(1), device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, padding], dim=1)
                
            # Create 2D attention mask
            attn_mask = attention_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
        else:
            # Create a full attention mask (all positions attend to all others)
            seq_len = sequence.size(1)
            attn_mask = torch.ones(batch_size, seq_len, seq_len, device=sequence.device)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            sequence = layer(sequence, attn_mask)
        if torch.isnan(sequence).any():
            print("NaN after transformer layers")
        
        # Global average pooling (considering mask)
        if attention_mask is not None:
            # Expanded mask for pooling with variable-length sequences
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(sequence)
            sequence = sequence * mask_expanded
            pooled = sequence.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            # Standard pooling for fixed-length sequences
            pooled = torch.mean(sequence, dim=1)

        # Apply layer normalization to pooled raw features
        raw_feat = self.layer_norm(pooled)  # [batch_size, embedding_dim]
        # Encode external features if provided
        feat = self.feature_encoder(x_feat) if (self.feature_encoder is not None and x_feat is not None) else None

        # Compute LDA reliability for CSP features if available
        if x_feat is not None:
            # Signed distance to LDA decision boundary
            lda_margin = torch.abs(x_feat @ self.lda_W.t() + self.lda_b)  # [B,1]
            # Compute reliability: use global margin stats if available, else batch z-score
            if hasattr(self, 'margin_mean') and hasattr(self, 'margin_std'):
                rel = (lda_margin - self.margin_mean) / (self.margin_std + 1e-6)
            else:
                # Use population std (unbiased=False) to avoid NaN on single-sample batches
                rel = (lda_margin - lda_margin.mean(dim=0, keepdim=True)) / (lda_margin.std(dim=0, keepdim=True, unbiased=False) + 1e-6)
        else:
            rel = torch.zeros(batch_size, 1, device=x_raw.device)
        # Fuse raw and feature streams via gated fusion with reliability
        fused, _ = self.fusion(raw_feat, feat, rel)  # [batch_size, fusion_dim]
        
        # Network1 hidden features and logits
        hidden1 = self.classifier_body(fused)
        # Network2 hidden features and logits
        hidden2 = self.adapter_head(fused)
        # Combine hidden features for combiner head
        combined_hidden = torch.cat([hidden1, hidden2], dim=1)
        # Network3 final prediction
        logits3 = self.combiner_head(combined_hidden)
        return logits3

