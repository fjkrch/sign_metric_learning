"""
Temporal Transformer encoder for landmark sequences.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding.

    Args:
        d_model: Feature dimensionality.
        max_len: Maximum sequence length.
        dropout: Dropout rate.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding.

        Args:
            x: ``(B, T, D)``

        Returns:
            ``(B, T, D)`` with positional encoding added.
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TemporalTransformerEncoder(nn.Module):
    """Transformer encoder operating on temporal landmark sequences.

    Input:
        ``(B, T, 21, 3)`` raw landmarks → projected to ``(B, T, d_model)``
        ``(B, T, 210)`` pairwise distances → projected to ``(B, T, d_model)``

    Output:
        ``(B, embedding_dim)`` pooled embedding.

    Args:
        input_dim: Per-frame feature dimension (21*3=63 or 210).
        embedding_dim: Output embedding size.
        num_heads: Number of attention heads.
        num_layers: Number of Transformer encoder layers.
        dim_feedforward: FFN hidden dimension.
        dropout: Dropout rate.
        max_len: Maximum sequence length for positional encoding.
    """

    def __init__(
        self,
        input_dim: int = 63,
        embedding_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        # d_model must be divisible by num_heads
        self.d_model = (embedding_dim // num_heads) * num_heads
        if self.d_model < num_heads:
            self.d_model = num_heads

        self.input_proj = nn.Linear(input_dim, self.d_model)
        self.pos_enc = PositionalEncoding(self.d_model, max_len=max_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pool_proj = nn.Linear(self.d_model, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(B, T, 21, 3)`` or ``(B, T, D)``.

        Returns:
            ``(B, embedding_dim)`` embedding.
        """
        if x.dim() == 4:
            B, T, V, C = x.shape
            x = x.reshape(B, T, V * C)  # (B, T, 63)
        # x: (B, T, D)
        x = self.input_proj(x)   # (B, T, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)  # (B, T, d_model)

        # Global average pooling over time
        x = x.mean(dim=1)         # (B, d_model)
        x = self.pool_proj(x)     # (B, embedding_dim)
        x = self.norm(x)
        return x


def build_transformer_encoder(cfg: dict, representation: str = "raw") -> TemporalTransformerEncoder:
    """Factory to build a Temporal Transformer encoder from config.

    Args:
        cfg: Full config dict.
        representation: ``'raw'`` | ``'pairwise'`` | ``'graph'``.

    Returns:
        Configured ``TemporalTransformerEncoder``.
    """
    if representation == "pairwise":
        input_dim = 210
    else:
        input_dim = 21 * 3  # 63

    t_cfg = cfg["model"]["transformer"]
    return TemporalTransformerEncoder(
        input_dim=input_dim,
        embedding_dim=cfg["model"]["embedding_dim"],
        num_heads=t_cfg["num_heads"],
        num_layers=t_cfg["num_layers"],
        dim_feedforward=t_cfg["dim_feedforward"],
        dropout=t_cfg["dropout"],
    )
