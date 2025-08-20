import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size, embed_dim, beta=0.25):
        super().__init__()
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.beta = beta
        self.codebook = nn.Embedding(codebook_size, embed_dim)
        nn.init.uniform_(self.codebook.weight, -1.0/embed_dim, 1.0/embed_dim)

    def forward(self, z_e):
        B, N, D = z_e.shape
        z_flat = z_e.view(-1, D)                           # [B*N, D]
        dist = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(1, keepdim=True).t()
        )
        codes = torch.argmin(dist, dim=1)                  # [B*N]
        z_q = self.codebook(codes).view(B, N, D)           # [B, N, D]
        codes = codes.view(B, N)

        loss_vq = F.mse_loss(z_q.detach(), z_e) + self.beta * F.mse_loss(z_q, z_e.detach())
        z_q = z_e + (z_q - z_e).detach()
        perp = torch.exp(-torch.sum(F.softmax(-dist, dim=1) * F.log_softmax(-dist, dim=1), dim=1).mean())

        return z_q, codes, loss_vq, perp

class VectorQuantizerEMA(nn.Module):
    """
    Optional EMA version (not required at first).
    """
    def __init__(self, codebook_size, embed_dim, decay=0.99, eps=1e-5):
        super().__init__()
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.decay = decay
        self.eps = eps

        self.embedding = nn.Embedding(codebook_size, embed_dim)
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed_avg", torch.zeros(codebook_size, embed_dim))
        self.embedding.weight.data.normal_()

    def forward(self, z_e):
        raise NotImplementedError("EMA quantizer can be added later")
