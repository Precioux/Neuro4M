import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

from fmri_numpy import make_multi_loader
from autoencoder1d import ViT_VQ_AE_1D
from trainer1d import train_one_epoch, evaluate

def parse_args():
    p = argparse.ArgumentParser("Train ViT-VQ-AE (1D) and export discrete tokens.")
    
    # Data
    p.add_argument("--paths", type=str, required=True,
                   help="Comma-separated *_betas_visual_only.npy files.")
    p.add_argument("--zscore", action="store_true")
    p.add_argument("--workers", type=int, default=4)

    # Model
    p.add_argument("--patch_len", type=int, default=64)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--depth_enc", type=int, default=6)
    p.add_argument("--depth_dec", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--codebook_size", type=int, default=1024)
    p.add_argument("--vq_beta", type=float, default=0.25)

    # Train
    p.add_argument("--save_dir", type=str, default="outputs/vitvq_train")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad_clip", type=float, default=0.0)

    # Export
    p.add_argument("--export_dir", type=str, default=None)
    p.add_argument("--export_batch_size", type=int, default=256)
    p.add_argument("--export_shard_size", type=int, default=10000)
    p.add_argument("--save_zq", action="store_true")
    p.add_argument("--save_mse", action="store_true")

    return p.parse_args()

@torch.no_grad()
def export_tokens(model, dataset, args, feature_dim):
    os.makedirs(args.export_dir, exist_ok=True)
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=args.export_batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    # Save codebook
    codebook = model.vq.codebook.weight.detach().cpu().numpy()
    np.save(os.path.join(args.export_dir, "codebook.npy"), codebook)

    K = int(args.codebook_size)
    codes_dtype = np.uint8 if K <= 256 else (np.uint16 if K <= 65535 else np.int32)

    shard_size = int(args.export_shard_size)
    shard_idx = 0
    buf_codes, buf_zq, buf_mse = [], [], []
    total, sum_perp, sum_mse = 0, 0.0, 0.0

    for xb in loader:
        xb = xb.to(device, non_blocking=True)
        xb_flat = xb.view(xb.size(0), -1)
        with autocast(enabled=torch.cuda.is_available()):
            xh, z_e, z_q, codes, loss_vq, perp = model(xb_flat)

        B = xb_flat.size(0)
        total += B
        sum_perp += float(perp) * B
        codes_np = codes.cpu().numpy().astype(codes_dtype)
        buf_codes.append(codes_np)

        if args.save_zq:
            buf_zq.append(z_q.cpu().numpy().astype(np.float32))
        if args.save_mse:
            mse = ((xh - xb_flat) ** 2).mean(dim=1).cpu().numpy()
            buf_mse.append(mse)
            sum_mse += float(mse.sum())
        else:
            sum_mse += float(((xh - xb_flat) ** 2).mean().item()) * B

        if sum(x.shape[0] for x in buf_codes) >= shard_size:
            np.save(os.path.join(args.export_dir, f"codes_{shard_idx:05d}.npy"), np.concatenate(buf_codes, axis=0))
            buf_codes.clear()
            if args.save_zq and buf_zq:
                np.save(os.path.join(args.export_dir, f"z_q_{shard_idx:05d}.npy"), np.concatenate(buf_zq, axis=0))
                buf_zq.clear()
            if args.save_mse and buf_mse:
                np.save(os.path.join(args.export_dir, f"mse_{shard_idx:05d}.npy"), np.concatenate(buf_mse, axis=0))
                buf_mse.clear()
            shard_idx += 1

    if buf_codes:
        np.save(os.path.join(args.export_dir, f"codes_{shard_idx:05d}.npy"), np.concatenate(buf_codes, axis=0))
    if args.save_zq and buf_zq:
        np.save(os.path.join(args.export_dir, f"z_q_{shard_idx:05d}.npy"), np.concatenate(buf_zq, axis=0))
    if args.save_mse and buf_mse:
        np.save(os.path.join(args.export_dir, f"mse_{shard_idx:05d}.npy"), np.concatenate(buf_mse, axis=0))

    stats = {
        "N": int(len(dataset)),
        "F": feature_dim,
        "patch_len": int(args.patch_len),
        "num_patches": int(feature_dim // args.patch_len),
        "embed_dim": int(args.embed_dim),
        "codebook_size": K,
        "avg_perplexity": float(sum_perp / max(total, 1)),
        "avg_mse": float(sum_mse / max(total, 1))
    }
    with open(os.path.join(args.export_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # Load data
    paths = [s.strip() for s in args.paths.split(",")]
    full_ds, full_loader = make_multi_loader(
        npy_paths=paths,
        batch_size=args.batch_size,
        num_workers=args.workers,
        patch_size=args.patch_len,
        zscore=args.zscore,
        shuffle=True,
        pin_memory=True
    )

    val_n = min(10000, len(full_ds))
    val_ds = torch.utils.data.Subset(full_ds, range(val_n))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    feature_dim = full_ds.datasets[0].f
    assert (feature_dim % args.patch_len) == 0, "feature_dim must be divisible by patch_len"

    model = ViT_VQ_AE_1D(
        feature_dim=feature_dim,
        patch_len=args.patch_len,
        embed_dim=args.embed_dim,
        depth_enc=args.depth_enc,
        depth_dec=args.depth_dec,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        codebook_size=args.codebook_size,
        vq_beta=args.vq_beta
    ).to(device)

    opt = AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    best = float("inf")
    best_path = os.path.join(args.save_dir, "best.pt")

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(
            model, full_loader, opt, device, scaler,
            codebook_size=args.codebook_size,
            scheduler=None,
            grad_clip=args.grad_clip
        )
        ev = evaluate(
            model, val_loader, device,
            codebook_size=args.codebook_size,
            save_dir=args.save_dir,
            epoch=epoch
        )

        print(f"epoch {epoch:03d} | "
              f"train MSE {tr['train_mse']:.6f} | train VQ {tr['train_vq']:.6f} | "
              f"val MSE {ev['val_mse']:.6f} | R2 {ev['val_r2']:.4f} | "
              f"perplexity {ev['val_perp']:.2f} | used {ev['used_codes']} | dead {ev['dead_codes']}")

        torch.save(model.state_dict(), os.path.join(args.save_dir, "last.pt"))
        if ev["val_mse"] < best:
            best = ev["val_mse"]
            torch.save(model.state_dict(), best_path)

    # Optional export
    if args.export_dir is not None:
        model.load_state_dict(torch.load(best_path, map_location="cpu"))
        model.eval()
        export_tokens(model, full_ds, args, feature_dim=feature_dim)

if __name__ == "__main__":
    main()
