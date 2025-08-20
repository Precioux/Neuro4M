import os
import json
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader


def _memmap_from_meta(npy_path: str) -> Tuple[np.memmap, int, int]:
    base = os.path.basename(npy_path)
    if "_betas_visual_only.npy" not in base:
        raise ValueError(f"Unsupported filename (expected '*_betas_visual_only.npy'): {base}")
    subj = base.split("_betas_visual_only.npy")[0]
    meta_path = os.path.join(os.path.dirname(npy_path), f"{subj}_betas_visual_only.meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing meta JSON for {base}: {meta_path}")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    n_trials = int(meta["total_trials"])
    n_feats = int(meta["n_visual_total"])
    mm = np.memmap(npy_path, dtype=np.float32, mode="r", shape=(n_trials, n_feats))
    return mm, n_trials, n_feats


def _patchify_row(x: np.ndarray, patch_size: int) -> np.ndarray:
    # x: [F] float32. Pad with zeros to a multiple of patch_size, then reshape to [L, P].
    F = x.shape[0]
    P = int(patch_size)
    rem = F % P
    if rem != 0:
        pad = P - rem
        x = np.pad(x, (0, pad), mode="constant", constant_values=0.0)
    L = x.shape[0] // P
    return x.reshape(L, P)


class VisualNSDPatched(Dataset):
    """
    Read-only memmap dataset. Each item returns a tensor of shape [L, P].
    Robust to NaN/Inf values; applies optional row-wise z-score.
    """
    def __init__(self, npy_path: str, patch_size: int, zscore: bool = False, eps: float = 1e-6):
        self.npy_path = npy_path
        self.mm, self.n, self.f = _memmap_from_meta(npy_path)
        self.patch_size = int(patch_size)
        self.zscore = bool(zscore)
        self.eps = float(eps)

        # Precompute sequence length after padding, for quick checks
        rem = self.f % self.patch_size
        padded_f = self.f if rem == 0 else (self.f + (self.patch_size - rem))
        self.seq_len = padded_f // self.patch_size  # L

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> torch.Tensor:
        row = np.asarray(self.mm[idx], dtype=np.float32, order="C")  # [F]

        # Clean invalids before normalization
        row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)

        if self.zscore:
            m = np.nanmean(row, dtype=np.float32)
            s = np.nanstd(row, dtype=np.float32)
            row = (row - m) / (s + self.eps)

        # Clean again after normalization
        row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)

        patches = _patchify_row(row, self.patch_size)  # [L, P]
        return torch.from_numpy(patches)  # float32


def make_multi_loader(
    npy_paths: List[str],
    batch_size: int,
    num_workers: int,
    patch_size: int,
    session_ids_path: Optional[str] = None,  # kept for API compatibility
    zscore: bool = False,
    pin_memory: bool = True,
    shuffle: bool = True,
) -> Tuple[ConcatDataset, DataLoader]:
    """
    Returns (dataset, dataloader). Each batch has shape [B, L, P].
    """
    datasets = [VisualNSDPatched(p, patch_size=patch_size, zscore=zscore) for p in npy_paths]
    ds = ConcatDataset(datasets)

    def _worker_init_fn(_):
        pass

    dl = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=False,
        worker_init_fn=_worker_init_fn,
    )
    return ds, dl
