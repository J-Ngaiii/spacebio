import torch
from torch import nn
from performer_pytorch import Performer as PerformerEncoder

from src.models.gene_embedding import GeneExpressionEmbedding


class Performer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        input_channels: int = 1,
        d_model: int = 128,
        depth: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        metadata_cardinalities: dict[str, int] | None = None,
    ):
        super().__init__()

        if input_channels != 1:
            raise ValueError("Expected one value per gene")

        if d_model % heads != 0:
            raise ValueError("d_model must be divisible by heads")

        self.embedding = GeneExpressionEmbedding(
            num_genes=input_dim,
            d_model=d_model,
            metadata_cardinalities=metadata_cardinalities or {},
        )

        self.encoder = PerformerEncoder(
            dim=d_model,
            depth=depth,
            heads=heads,
            dim_head=d_model // heads,
            causal=False,
            ff_dropout=dropout,
            attn_dropout=dropout,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )

    def forward(self, x, metadata):
        if x.ndim == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)

        x = self.embedding(x, metadata)
        x = self.encoder(x)
        x = x.mean(dim=1)

        return self.head(x)