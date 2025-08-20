import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, n_codes, embedding_dim, beta=0.25, use_cosine=False):
        super().__init__()
        self.n_codes = n_codes
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.use_cosine = use_cosine

        self.embedding = nn.Parameter(torch.randn(n_codes, embedding_dim))

    def forward(self, z):
        B, N, D = z.shape
        flat_z = z.view(-1, D)

        if self.use_cosine:
            emb = F.normalize(self.embedding, dim=1)
            flat_z = F.normalize(flat_z, dim=1)
        else:
            emb = self.embedding

        dist = (
            torch.sum(flat_z ** 2, dim=1, keepdim=True)
            - 2 * torch.matmul(flat_z, emb.t())
            + torch.sum(emb ** 2, dim=1)
        )  # [B*N, K]

        codes = torch.argmin(dist, dim=1)  # [B*N]
        z_q = emb[codes].view(B, N, D)

        z_q = z + (z_q - z).detach()
        loss = self.beta * F.mse_loss(z_q, z.detach())

        one_hot = F.one_hot(codes, num_classes=self.n_codes).float()
        avg_probs = one_hot.mean(dim=0)
        perp = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q, codes.view(B, N), loss, perp
