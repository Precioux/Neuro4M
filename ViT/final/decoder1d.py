import torch
import torch.nn as nn
from vit1d_encoder import TransformerBlock

class Decoder1D(nn.Module):
    def __init__(self, patch_len, embed_dim, depth, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.layers = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop, attn_drop)
            for _ in range(depth)
        ])
        self.out_proj = nn.Linear(embed_dim, patch_len)

    def forward(self, x: torch.Tensor, meta: dict) -> torch.Tensor:
        # x: [B, L, D]
        x = self.layers(x)              # [B, L, D]
        x = self.out_proj(x)            # [B, L, P]
        x = x.flatten(1, 2)             # [B, F]
        return x
