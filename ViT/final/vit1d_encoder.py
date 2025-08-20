import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, dropout=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=attn_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLPBlock(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x):
        x_res = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + x_res

        x_res = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + x_res
        return x

class ViT1DEncoder(nn.Module):
    def __init__(
        self,
        patch_len,
        seq_len,
        dim=512,
        depth=6,
        heads=8,
        mlp_ratio=4.0,
        drop=0.0,
        token_drop=0.0,
        pre_norm=True,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.seq_len = seq_len
        self.token_drop = nn.Dropout(token_drop)
        self.pre_norm = pre_norm

        self.input_proj = nn.Linear(patch_len, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, dim) * 0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_ratio, drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim) if pre_norm else nn.Identity()

    def forward(self, x):
        # x: [B, L, P]
        x = self.input_proj(x) + self.pos_embed  # [B, L, D]
        x = self.token_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x, {"patch_len": self.patch_len, "seq_len": self.seq_len}
