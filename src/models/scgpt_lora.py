"""scGPT backbone loading, LoRA wrapping and prediction head.

Used by src/training/train/scgpt_lora_train.py. Written against scGPT's TransformerModel
API from memory and not run: check the "loaded N/M pretrained tensors" line first.
"""
import json
from pathlib import Path

import torch
import torch.nn as nn

from src.training.train_conf.scgpt_lora_config import scgpt_lora_train_config


def _load_backbone(checkpoint_dir: str):
    from scgpt.model import TransformerModel
    from scgpt.tokenizer import GeneVocab

    d = Path(checkpoint_dir)
    vocab = GeneVocab.from_file(d / "vocab.json")
    args = json.load(open(d / "args.json"))
    model = TransformerModel(
        ntoken=len(vocab),
        d_model=args["embsize"],
        nhead=args["nheads"],
        d_hid=args["d_hid"],
        nlayers=args["nlayers"],
        vocab=vocab,
        dropout=args["dropout"],
        pad_token="<pad>",
        pad_value=args.get("pad_value", -2),
        n_input_bins=args.get("n_bins", 51),
        use_fast_transformer=False,  # avoids the flash-attn dependency
    )
    state = torch.load(d / "best_model.pt", map_location="cpu")
    # Flash-attn checkpoints store fused qkv under Wqkv; remap for torch attention.
    state = {
        k.replace("self_attn.Wqkv.weight", "self_attn.in_proj_weight")
         .replace("self_attn.Wqkv.bias", "self_attn.in_proj_bias"): v
        for k, v in state.items()
    }
    own = model.state_dict()
    ok = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    own.update(ok)
    model.load_state_dict(own)
    print(f"loaded {len(ok)}/{len(own)} pretrained tensors")  # should be nearly all of them
    return model, vocab, args["embsize"]


class ScGPTRegressionHead(nn.Module):
    """scGPT backbone + small MLP head on the <cls> embedding."""

    def __init__(self, backbone, d_model, cls_id, pad_value, n_out, hidden, dropout):
        super().__init__()
        self.backbone = backbone
        self.cls_id = cls_id
        self.pad_value = pad_value
        # Standardizes the <cls> embedding (set per fit from the training samples).
        self.register_buffer("emb_mean", torch.zeros(d_model))
        self.register_buffer("emb_std", torch.ones(d_model))
        self.head = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_out)
        )

    def embed(self, gene_ids, values):
        B = gene_ids.size(0)
        cls = torch.full((B, 1), self.cls_id, dtype=gene_ids.dtype, device=gene_ids.device)
        ids = torch.cat([cls, gene_ids], dim=1)
        vals = torch.cat(
            [torch.full((B, 1), float(self.pad_value), device=values.device), values.float()], dim=1
        )
        no_pad = torch.zeros_like(ids, dtype=torch.bool)  # every sample shares one gene set
        # NOTE: if the checkpoint uses a "category" value encoder instead of "continuous",
        # vals must be long instead of float.
        out = self.backbone._encode(ids, vals, src_key_padding_mask=no_pad)
        return out[:, 0, :]

    def set_embedding_stats(self, mean, std):
        self.emb_mean.copy_(mean)
        self.emb_std.copy_(std)

    def forward(self, gene_ids, values):
        return self.head((self.embed(gene_ids, values) - self.emb_mean) / self.emb_std)


def build_scgpt_lora(config: scgpt_lora_train_config, n_out: int) -> ScGPTRegressionHead:
    backbone, vocab, d_model = _load_backbone(config.checkpoint_dir)
    for p in backbone.parameters():
        p.requires_grad = False
    if config.use_lora:
        from peft import LoraConfig, get_peft_model

        # NOTE: nn.MultiheadAttention keeps qkv as one raw parameter, so LoRA attaches to the
        # attention output projection and the feed-forward layers rather than q/k/v.
        backbone = get_peft_model(
            backbone,
            LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=["out_proj", "linear1", "linear2"],
            ),
        )
    model = ScGPTRegressionHead(
        backbone, d_model, vocab["<cls>"], config.pad_value, n_out, config.head_hidden_dim, config.head_dropout
    )
    return model