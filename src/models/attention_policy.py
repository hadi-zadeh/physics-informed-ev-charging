"""
attention_policy.py
Multi-Head Self-Attention (MHSA) Policy Architecture.
Temporal attention encoder with position encoding and deep MLP decision head.
Total trainable parameters: 990,917.
"""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 50, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TemporalAttentionExtractor(nn.Module):
    def __init__(self, in_features: int = 3, d_model: int = 64, n_heads: int = 4,
                 seq_len: int = 24, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.input_proj = nn.Linear(in_features, d_model)
        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=seq_len + 5, dropout=dropout)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads,
                                          dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.pos_enc(h)
        attn_out, _ = self.attn(h, h, h)
        h = self.norm1(h + self.dropout(attn_out))
        ff_out = self.ff(h)
        h = self.norm2(h + self.dropout(ff_out))
        return h.reshape(h.size(0), -1)


class AttentionPolicy(nn.Module):
    """
    Transformer/Attention-based policy network for EV charging control.
    Uses multi-head self-attention over rolling 24-step horizons.
    """
    def __init__(self, state_dim: int = 74, action_dim: int = 1,
                 e_min: float = -7.0, e_max: float = 7.0,
                 d_model: int = 64, n_heads: int = 4, seq_len: int = 24,
                 hidden1: int = 512, hidden2: int = 256, hidden3: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout

        self.attention = TemporalAttentionExtractor(
            in_features=3, d_model=d_model, n_heads=n_heads,
            seq_len=seq_len, dropout=dropout
        )

        fusion_in = seq_len * d_model + 2
        self.fusion_norm = nn.LayerNorm(fusion_in)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, hidden1),
            nn.LayerNorm(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden3),
            nn.LayerNorm(hidden3),
            nn.ReLU(),
            nn.Linear(hidden3, action_dim)
        )

        self.act = nn.Hardtanh(min_val=e_min, max_val=e_max)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        price = states[:, 0:24].unsqueeze(-1)
        pv = states[:, 24:48].unsqueeze(-1)
        load = states[:, 48:72].unsqueeze(-1)
        series_3d = torch.cat([price, pv, load], dim=-1)

        attn_features = self.attention(series_3d)
        scalars = states[:, 72:74]
        fused = torch.cat([attn_features, scalars], dim=1)

        h = self.fusion(self.fusion_norm(fused))
        out = self.mlp(h)
        return self.act(out)
