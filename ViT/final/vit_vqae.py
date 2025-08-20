import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4, dropout=0.0, pre_norm=True):
        super().__init__()
        self.pre_norm = pre_norm
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        if self.pre_norm:
            y, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
            x = x + y
            x = x + self.mlp(self.ln2(x))
        else:
            y, _ = self.attn(x, x, x, need_weights=False)
            x = self.ln1(x + y)
            x = self.ln2(x + self.mlp(x))
        return x

class TokenDropout(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p
    def forward(self, x):
        if not self.training or self.p <= 0: return x
        B, L, D = x.shape
        mask = (torch.rand(B, L, device=x.device) > self.p).float().unsqueeze(-1)
        return x * mask

class ViTVQAE1D(nn.Module):
    def __init__(self, seq_len, code_dim, d_model=512, depth=10, heads=8, mlp_ratio=4,
                 dropout=0.1, token_dropout=0.05, pre_norm=True):
        super().__init__()
        self.seq_len = seq_len
        self.code_dim = code_dim

        self.enc_in = nn.LazyLinear(d_model)
        self.dec_out = nn.LazyLinear(out_features=None)

        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model) / math.sqrt(d_model))
        self.tdrop = TokenDropout(token_dropout)
        self.enc_ln = nn.LayerNorm(d_model)
        self.dec_ln = nn.LayerNorm(d_model)

        self.enc_blocks = nn.ModuleList([
            TransformerBlock(d_model, heads, mlp_ratio, dropout, pre_norm) for _ in range(depth)
        ])
        self.to_code = nn.Linear(d_model, code_dim)
        self.from_code = nn.Linear(code_dim, d_model)

        self.dec_blocks = nn.ModuleList([
            TransformerBlock(d_model, heads, mlp_ratio, dropout, pre_norm) for _ in range(max(2, depth // 2))
        ])

    def _lazy_set_dec_out(self, x):
        if isinstance(self.dec_out, nn.LazyLinear):
            in_features = x.shape[-1]
            self.dec_out = nn.Linear(self.from_code.out_features, in_features)

    def encode(self, x):  # x: [B, L, F]
        B, L, _ = x.shape
        assert L == self.seq_len, f"seq_len mismatch: {L} != {self.seq_len}"
        h = self.enc_in(x) + self.pos_emb
        h = self.tdrop(h)
        for blk in self.enc_blocks:
            h = blk(h)
        h = self.enc_ln(h)
        z_e = self.to_code(h)
        return z_e

    def decode(self, z_q):  # z_q: [B, L, code_dim]
        h = self.from_code(z_q)
        for blk in self.dec_blocks:
            h = blk(h)
        h = self.dec_ln(h)
        x_hat = self.dec_out(h)
        return x_hat
