
import math
import torch
from torch import nn
from torch.nn import functional as F


class LinearAttention(nn.Module):
    """Noncausal, positive-feature linear attention."""

    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")

        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, length, dim = x.shape

        q, k, v = self.qkv(x).reshape(
            batch, length, 3, self.heads, self.head_dim
        ).unbind(dim=2)

        q, k, v = [
            t.transpose(1, 2) for t in (q, k, v)
        ]

        # Positive feature map; linear in sequence length.
        q = F.elu(q) + 1
        k = F.elu(k) + 1

        q = q * self.head_dim ** -0.25
        k = k * self.head_dim ** -0.25

        kv = torch.einsum("bhld,bhle->bhde", k, v)
        numerator = torch.einsum("bhld,bhde->bhle", q, kv)

        denominator = torch.einsum(
            "bhld,bhd->bhl", q, k.sum(dim=2)
        ).unsqueeze(-1)

        result = numerator / denominator.clamp_min(1e-6)

        result = result.transpose(1, 2).reshape(
            batch, length, dim
        )
        return self.dropout(self.out(result))


class PerformerBlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = LinearAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class Performer(nn.Module):
    """
    Generic supervised model.

    Input:
        (batch, features) or (batch, sequence, channels)

    Output:
        (batch, output_dim)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 128,
        depth: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        input_channels: int = 1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.input_channels = input_channels

        self.embedding = nn.Linear(input_channels, d_model)
        self.position = nn.Parameter(
            torch.randn(1, input_dim, d_model) * 0.02
        )

        self.blocks = nn.Sequential(*[
            PerformerBlock(d_model, heads, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(-1)

        if x.ndim != 3:
            raise ValueError("Expected 2D or 3D input")

        if x.shape[1] != self.input_dim:
            raise ValueError("Unexpected sequence length")

        if x.shape[2] != self.input_channels:
            raise ValueError("Unexpected input channels")

        x = self.embedding(x) + self.position
        x = self.blocks(x)
        x = self.norm(x).mean(dim=1)

        return self.head(x)
