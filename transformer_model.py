import math
import torch
import torch.nn as nn
from typing import Optional

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch, d_model)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        num_features: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        num_classes: int = 2,
        pooling: str = 'last',  # 'last' or 'mean'
    ):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False, # API simplification & masking: set batch_first=True to remove transposes and reduce shape bugs.
            activation='gelu',
            norm_first=True,
        )
        # If you later pad variable-length sequences, ensure src_key_padding_mask has shape (batch, seq_len). If you need strictly causal attention (no peeking within the window), provide a causal attn_mask.
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.pooling = pooling
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None):
        # x: (batch, seq_len, num_features)
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        x = x.transpose(0, 1)   # (seq_len, batch, d_model)
        x = self.positional_encoding(x)
        z = self.transformer(x, src_key_padding_mask=src_key_padding_mask)  # (seq_len, batch, d_model)
        if self.pooling == 'mean':
            h = z.mean(dim=0)  # (batch, d_model)
        else:
            h = z[-1]  # last token (batch, d_model)
        h = self.norm(h)
        logits = self.head(h)
        return logits