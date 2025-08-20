import torch
import torch.nn as nn

from vit1d_encoder import ViT1DEncoder
from decoder1d import Decoder1D
from quantizer_ema import VectorQuantizer

class Autoencoder1D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        codebook_size = cfg["codebook_size"]
        embed_dim = cfg["embedding_dim"]
        vit_cfg = cfg["vit"]
        vq_cfg = cfg.get("vq", {})

        self.encoder = ViT1DEncoder(
            patch_len=vit_cfg["patch_len"],
            seq_len=vit_cfg["seq_len"],
            dim=vit_cfg["dim"],
            depth=vit_cfg["depth"],
            heads=vit_cfg["heads"],
            mlp_ratio=vit_cfg.get("mlp_ratio", 4.0),
            drop=vit_cfg.get("dropout", 0.0),
            token_drop=vit_cfg.get("token_dropout", 0.0),
            pre_norm=vit_cfg.get("pre_norm", True),
        )

        self.vq = VectorQuantizer(
            codebook_size,
            embed_dim,
            beta=vq_cfg.get("beta_commit", 0.25),
            use_cosine=vq_cfg.get("use_cosine", False)
        )

        self.decoder = Decoder1D(
            patch_len=vit_cfg["patch_len"],
            seq_len=vit_cfg["seq_len"],
            dim=vit_cfg["dim"],
            depth=vit_cfg.get("depth_dec", 4),
            heads=vit_cfg["heads"],
            mlp_ratio=vit_cfg.get("mlp_ratio", 4.0),
            drop=vit_cfg.get("dropout", 0.0),
            pre_norm=vit_cfg.get("pre_norm", True),
        )

    def forward(self, x):
        z_e, meta = self.encoder(x)                 # [B, N, D]
        z_q, codes, loss_vq, perplexity = self.vq(z_e)  # [B, N, D]
        recon = self.decoder(z_q, meta)             # [B, F]
        return recon, z_e, z_q, codes, loss_vq, perplexity
