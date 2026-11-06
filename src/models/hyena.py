
from src.models.gene_embedding import GeneExpressionEmbedding


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

        # Gene identity + expression + metadata embeddings
        self.embedding = GeneExpressionEmbedding(
            num_genes=input_dim,
            d_model=d_model,
            metadata_cardinalities=metadata_cardinalities or {},
        )

        # Hyena backbone
        self.blocks = nn.Sequential(*[
            HyenaBlock(
                d_model, order, kernel_size, dropout
            )
            for _ in range(depth)
        ])

        # Protein abundance prediction head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x, metadata=None):
        # Accept (B, M) or (B, M, 1)
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
