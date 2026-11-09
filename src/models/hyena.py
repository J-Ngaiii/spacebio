
import torch
import torch.nn.functional as F
from torch import nn

from src.models.gene_embedding import GeneExpressionEmbedding


class HyenaOperator(nn.Module):
    """
    Gated long convolution over the gene sequence.

    Input/output:
        (batch, num_genes, d_model)
    """

    def __init__(
        self,
        d_model: int,
        order: int = 2,
        kernel_size: int = 129,
        dropout: float = 0.1,
    ):
        super().__init__()

        if order < 1:
            raise ValueError("order must be at least 1")
        if kernel_size < 1:
            raise ValueError("kernel_size must be positive")

        self.d_model = d_model
        self.order = order
        self.kernel_size = kernel_size

        # One signal and 'order' multiplicative gates.
        self.in_proj = nn.Linear(
            d_model,
            (order + 1) * d_model,
        )

        # Learnable convolution kernels, one per order and channel.
        self.filters = nn.Parameter(
            torch.randn(order, d_model, kernel_size) * 0.02
        )

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def fft_conv(self, x, kernel):
        """
        Depthwise causal convolution using FFT.

        x:      (B, L, D)
        kernel: (D, K)

        returns: (B, L, D)
        """
        B, L, D = x.shape
        K = kernel.shape[-1]

        fft_size = L + K - 1

        # FFT along sequence dimension.
        x_fft = torch.fft.rfft(
            x.transpose(1, 2).float(),
            n=fft_size,
            dim=-1,
        )

        k_fft = torch.fft.rfft(
            kernel.float(),
            n=fft_size,
            dim=-1,
        )

        # Broadcast kernel across the batch.
        y_fft = x_fft * k_fft.unsqueeze(0)

        y = torch.fft.irfft(
            y_fft,
            n=fft_size,
            dim=-1,
        )

        # Keep the causal portion with the original sequence length.
        y = y[..., :L]

        return y.transpose(1, 2).to(dtype=x.dtype)

    def forward(self, x):
        # x: (B, L, D)
        projected = self.in_proj(x)

        # Split into one signal and 'order' gates.
        parts = projected.chunk(self.order + 1, dim=-1)

        h = parts[0]
        gates = parts[1:]

        for i, gate in enumerate(gates):
            h = h * torch.sigmoid(gate)
            h = self.fft_conv(h, self.filters[i])

        return self.dropout(self.out_proj(h))


class HyenaBlock(nn.Module):
    """
    Pre-normalization residual Hyena block.

    Contains:
        1. Hyena operator with residual connection
        2. Feedforward network with residual connection
    """

    def __init__(
        self,
        d_model: int,
        order: int = 2,
        kernel_size: int = 129,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)

        self.hyena = HyenaOperator(
            d_model=d_model,
            order=order,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        self.norm2 = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.hyena(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Hyena(nn.Module):
    """
    Gene expression Hyena model with categorical metadata.

    Input:
        x: (batch, num_genes)
        metadata: dict[str, Tensor(batch,)]

    Output:
        (batch, output_dim)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 128,
        depth: int = 4,
        order: int = 2,
        kernel_size: int = 129,
        dropout: float = 0.1,
        input_channels: int = 1,
        metadata_cardinalities: dict[str, int] | None = None,
    ):
        super().__init__()

        if input_channels != 1:
            raise ValueError(
                "GeneExpressionEmbedding requires one expression "
                "value per gene"
            )

        self.input_dim = input_dim
        self.input_channels = input_channels

        # Gene identity + expression + metadata embeddings.
        self.embedding = GeneExpressionEmbedding(
            num_genes=input_dim,
            d_model=d_model,
            metadata_cardinalities=metadata_cardinalities or {},
        )

        # Hyena backbone.
        self.blocks = nn.Sequential(
            *[
                HyenaBlock(
                    d_model=d_model,
                    order=order,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        # Protein abundance prediction head.
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x, metadata=None):
        # Accept (B, M) or (B, M, 1).
        if x.ndim == 3:
            if x.shape[-1] != 1:
                raise ValueError("Expected one channel per gene")
            x = x.squeeze(-1)

        if x.ndim != 2:
            raise ValueError("Expected 2D gene expression input")

        if x.shape[1] != self.input_dim:
            raise ValueError("Gene count mismatch")

        if metadata is None:
            metadata = {}

        # (B, M) -> (B, M, d_model)
        x = self.embedding(x, metadata)

        # (B, M, d_model) -> (B, M, d_model)
        x = self.blocks(x)

        # (B, M, d_model) -> (B, d_model)
        x = self.norm(x).mean(dim=1)

        # (B, d_model) -> (B, output_dim)
        return self.head(x)
