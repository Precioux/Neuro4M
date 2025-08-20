import torch

def mse_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Mean squared error across all features per sample, averaged over batch.
    """
    return torch.mean((x_hat - x) ** 2)

@torch.no_grad()
def r2_score(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Coefficient of determination R^2 for reconstruction.
    R^2 = 1 - sum((x - x_hat)^2) / sum((x - mean(x))^2)
    Computed per batch and averaged.
    """
    ss_res = torch.sum((x - x_hat) ** 2, dim=1)
    mean_x = torch.mean(x, dim=1, keepdim=True)
    ss_tot = torch.sum((x - mean_x) ** 2, dim=1) + 1e-8
    r2 = 1.0 - ss_res / ss_tot
    return torch.mean(r2)
