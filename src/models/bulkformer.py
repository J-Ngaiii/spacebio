import torch.nn as nn
from bulkformer import BulkFormerModel


def load_pretrained_bulkformer(checkpoint: str = "zhfanrui/BulkFormer-37M") -> BulkFormerModel:
    """Download (first run only, then cached) and return a pretrained BulkFormer encoder."""
    return BulkFormerModel.from_pretrained(checkpoint)


def protein_head(embed_dim, output_dim, dropout=0.0):
    """Linear layer from one sample embedding to one value per protein."""
    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(embed_dim, output_dim)
    )


class BulkFormerRegressor(nn.Module):
    """Pretrained BulkFormer encoder plus a protein head, for fine-tuning the whole model."""

    def __init__(self, backbone, output_dim, dropout=0.0, mask_prob=None):
        super().__init__()

        self.backbone = backbone
        self.mask_prob = mask_prob  # fraction of vocabulary genes missing from the input
        # BulkFormer outputs dim + 3 numbers per gene.
        self.head = protein_head(backbone.config.dim + 3, output_dim, dropout=dropout)

    def forward(self, x):
        gene_emb = self.backbone(x, mask_prob=self.mask_prob, output_expr=False)  # (batch, genes, dim + 3)
        # Average genes into one embedding per sample, as extract_features(..., "sample_level") does.
        return self.head(gene_emb.mean(dim=1))
