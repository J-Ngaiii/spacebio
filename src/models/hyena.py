
import torch
from torch import nn
from torch.nn import functional as F


class HyenaOperator(nn.Module):
    def __init__(
        self,
        dim: int,
        order: int = 2,
        kernel_size: int = 129,
        dropout: float = 0.1,
    ):
        super().__init__()

        if order < 1 or kernel_size < 1:
            raise ValueError("order and kernel_size must be positive")

        self.order = order
        self.input_projection = nn.Linear(
            dim, (order + 1) * dim
        )

        self.kernels = nn.Parameter(
            torch.randn(order, dim, kernel_size) * 0.02
        )

        self.output_projection = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def fft_conv(x, kernel):
        length = x.shape[1]
        fft_size = length + kernel.shape[-1] - 1

        x_fft = torch.fft.rfft(
            x.transpose(1, 2).float(), n=fft_size
        )
        k_fft = torch.fft.rfft(
            kernel.float(), n=fft_size
        )

        output = torch.fft.irfft(
            x_fft * k_fft.unsqueeze(0),
            n=fft_size,
        )[..., :length]

        return output.transpose(1, 2).to(x.dtype)

    def forward(self, x):
        parts = self.input_projection(x).chunk(
            self.order + 1, dim=-1
        )

        state = parts[-1]

        for gate, kernel in zip(
            reversed(parts[:-1]), self.kernels
        ):
            state = self.fft_conv(state, kernel)
            state = state * torch.sigmoid(gate)

        return self.dropout(
            self.output_projection(state)
        )


class HyenaBlock(nn.Module):
    def __init__(self, dim, order, kernel_size, dropout):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.mixer = HyenaOperator(
            dim, order, kernel_size, dropout
        )
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class Hyena(nn.Module):
    """
    Generic supervised Hyena-style model.

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
        order: int = 2,
        kernel_size: int = 129,
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
            HyenaBlock(
                d_model, order, kernel_size, dropout
            )
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
