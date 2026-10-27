import torch
import torch.nn as nn


def load_pretrained_bulkformer(checkpoint_path: str, device: str = "cpu") -> nn.Module:
    """Return the pretrained BulkFormer encoder with weights loaded from checkpoint_path."""
    # TODO: install the BulkFormer repo, build its model class and load the checkpoint weights.
    raise NotImplementedError("Load the pretrained BulkFormer encoder.")


class BulkFormerRegressor(nn.Module):
    """Pretrained BulkFormer encoder plus a linear head that predicts one value per protein."""

    def __init__(self, backbone, embed_dim, output_dim, dropout=0.0, freeze_backbone=True):
        super().__init__()

        self.backbone = backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, output_dim)
        )

    def forward(self, x):
        # TODO: confirm the backbone returns one embedding per sample, (batch, embed_dim).
        # If it returns one per gene, (batch, n_genes, embed_dim), pool them here first, e.g. .mean(dim=1).
        return self.head(self.backbone(x))
