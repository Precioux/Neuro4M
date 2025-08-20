import torch
import torch.nn as nn

class Patchify1D(nn.Module):
    """
    Split 1D signal [B, F] into patches of length P.
    If input is already patchified as [B, N, P], it returns as-is.
    """
    def __init__(self, patch_len: int):
        super().__init__()
        self.patch_len = patch_len

    def forward(self, x: torch.Tensor):
        if x.ndim == 3:
            # Already in [B, N, P]
            return x
        elif x.ndim == 2:
            B, F = x.shape
            assert F % self.patch_len == 0, "F must be divisible by patch_len"
            N = F // self.patch_len
            x = x.view(B, N, self.patch_len)
            return x
        else:
            raise ValueError(f"Expected input shape [B, F] or [B, N, P], got {x.shape}")

class Unpatchify1D(nn.Module):
    """
    Reconstruct [B, N, P] back to [B, F].
    """
    def __init__(self, patch_len: int):
        super().__init__()
        self.patch_len = patch_len

    def forward(self, x: torch.Tensor):
        B, N, P = x.shape
        return x.view(B, N * P)
