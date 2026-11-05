import torch
from torch import nn


class GeneExpressionEmbedding(nn.Module):
    def __init__(
        self,
        num_genes: int,
        d_model: int,
        metadata_cardinalities: dict[str, int],
    ):
        super().__init__()

        self.num_genes = num_genes

        self.gene_embedding = nn.Embedding(
            num_genes, d_model
        )

        self.expression_embedding = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
        )

        self.metadata_embeddings = nn.ModuleDict({
            name: nn.Embedding(
                cardinality,
                d_model,
                padding_idx=0,
            )
            for name, cardinality in metadata_cardinalities.items()
        })

    def forward(self, x, metadata):
        # x: (B, M)
        # metadata: dict[str, tensor(B,)]

        B, M = x.shape

        if M != self.num_genes:
            raise ValueError("Gene count mismatch")

        gene_ids = torch.arange(M, device=x.device)

        gene_emb = self.gene_embedding(gene_ids)
        expression_emb = self.expression_embedding(
            x.unsqueeze(-1)
        )

        h = gene_emb.unsqueeze(0) + expression_emb

        for name, embedding in self.metadata_embeddings.items():
            condition_emb = embedding(metadata[name])
            h = h + condition_emb.unsqueeze(1)

        return h