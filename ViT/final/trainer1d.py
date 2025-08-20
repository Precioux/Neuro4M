import os
import torch
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Tuple, Optional

from metrics1d import mse_loss, r2_score


def count_code_usage(codes: torch.Tensor, codebook_size: int) -> Tuple[int, int]:
    hist = torch.bincount(codes.view(-1), minlength=codebook_size)
    used = int((hist > 0).sum().item())
    dead = int((hist == 0).sum().item())
    return used, dead


def ensure_flat(x: torch.Tensor) -> torch.Tensor:
    # Ensure x is [B, F]
    return x.view(x.size(0), -1) if x.dim() > 2 else x


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    scaler: GradScaler,
    codebook_size: int,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    grad_clip: float = 0.0,
) -> Dict[str, float]:
    model.train()
    total_rec, total_vq, total = 0.0, 0.0, 0

    for x in loader:
        x = x.to(device, non_blocking=True)
        x = ensure_flat(x)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=torch.cuda.is_available()):
            x_hat, z_e, z_q, codes, loss_vq, perp = model(x)
            x_hat = ensure_flat(x_hat)
            loss_rec = mse_loss(x_hat, x)
            loss = loss_rec + loss_vq

        scaler.scale(loss).backward()

        if grad_clip and grad_clip > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        bsz = x.size(0)
        total_rec += float(loss_rec.item()) * bsz
        total_vq += float(loss_vq.item()) * bsz
        total += bsz

    return {
        "train_mse": total_rec / max(total, 1),
        "train_vq": total_vq / max(total, 1),
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    codebook_size: int,
    save_dir: str,
    epoch: int
) -> Dict[str, float]:
    model.eval()
    sum_mse, sum_r2, sum_perp, n = 0.0, 0.0, 0.0, 0
    agg_codes = []
    first_batch = True

    for x in loader:
        x = x.to(device, non_blocking=True)
        x = ensure_flat(x)

        x_hat, z_e, z_q, codes, loss_vq, perp = model(x)
        x_hat = ensure_flat(x_hat)

        bsz = x.size(0)
        sum_mse += float(mse_loss(x_hat, x).item()) * bsz
        sum_r2 += float(r2_score(x_hat, x).item()) * bsz
        sum_perp += float(perp) * bsz
        n += bsz

        agg_codes.append(codes.cpu())

        if first_batch:
            first_batch = False
            os.makedirs(save_dir, exist_ok=True)
            torch.save(
                {
                    "z_e": z_e[:32].float().cpu(),
                    "z_q": z_q[:32].float().cpu(),
                    "codes": codes[:32].int().cpu(),
                },
                os.path.join(save_dir, f"samples_epoch{epoch:03d}.pt")
            )

    codes_all = torch.cat(agg_codes, dim=0) if len(agg_codes) else torch.empty(0, dtype=torch.long)
    used, dead = count_code_usage(codes_all, codebook_size) if codes_all.numel() > 0 else (0, codebook_size)

    return {
        "val_mse": sum_mse / max(n, 1),
        "val_r2": sum_r2 / max(n, 1),
        "val_perp": sum_perp / max(n, 1),
        "used_codes": used,
        "dead_codes": dead,
    }
