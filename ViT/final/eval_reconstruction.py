import os, json, argparse, numpy as np, torch
from torch.cuda.amp import autocast

from fmri_numpy import make_multi_loader, VisualNSDPatched
from autoencoder1d import ViT_VQ_AE_1D


@torch.no_grad()
def mse_per_sample(xh, x): return ((xh - x) ** 2).mean(dim=1)

@torch.no_grad()
def r2_per_sample(xh, x):
    mu = x.mean(dim=1, keepdim=True)
    ss_res = ((x - xh) ** 2).sum(dim=1)
    ss_tot = ((x - mu) ** 2).sum(dim=1) + 1e-12
    return 1.0 - ss_res / ss_tot

def parse_args():
    p = argparse.ArgumentParser("Evaluate reconstruction: direct or from saved codes.")
    p.add_argument("--paths", type=str, required=True)  # comma-separated .npy files
    p.add_argument("--patch_len", type=int, required=True)
    p.add_argument("--embed_dim", type=int, required=True)
    p.add_argument("--depth_enc", type=int, default=6)
    p.add_argument("--depth_dec", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--codebook_size", type=int, default=1024)
    p.add_argument("--vq_beta", type=float, default=0.25)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="outputs/eval")
    p.add_argument("--mode", type=str, default="direct", choices=["direct", "from_codes"])
    p.add_argument("--codes_dir", type=str, default=None)
    p.add_argument("--save_samples", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--shard_limit", type=int, default=None)
    p.add_argument("--zscore", action="store_true")
    return p.parse_args()

def list_code_shards(tokens_dir):
    files = [f for f in os.listdir(tokens_dir) if f.startswith("codes_") and f.endswith(".npy")]
    files.sort()
    return [os.path.join(tokens_dir, f) for f in files]

def make_meta(F, P): 
    return {"orig_len": int(F), "pad_len": 0, "num_patches": int(F // P)}


@torch.no_grad()
def eval_direct(args):
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths = [s.strip() for s in args.paths.split(",")]
    dataset, loader = make_multi_loader(
        npy_paths=paths,
        batch_size=args.batch_size,
        num_workers=args.workers,
        patch_size=args.patch_len,
        zscore=args.zscore,
        shuffle=False,
        pin_memory=True,
    )
    feature_dim = dataset.datasets[0].f

    model = ViT_VQ_AE_1D(
        feature_dim=feature_dim, patch_len=args.patch_len, embed_dim=args.embed_dim,
        depth_enc=args.depth_enc, depth_dec=args.depth_dec, num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio, codebook_size=args.codebook_size, vq_beta=args.vq_beta
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()

    all_mse, all_r2 = [], []
    saved = False
    for xb in loader:
        xb = xb.to(device)
        xb_flat = xb.view(xb.size(0), -1)
        with autocast(enabled=torch.cuda.is_available()):
            xh, *_ = model(xb_flat)
        all_mse.append(mse_per_sample(xh, xb_flat).cpu())
        all_r2.append(r2_per_sample(xh, xb_flat).cpu())
        if not saved and args.save_samples > 0:
            k = min(args.save_samples, xb.size(0))
            np.save(os.path.join(args.save_dir, "orig_samples.npy"), xb_flat[:k].cpu().numpy())
            np.save(os.path.join(args.save_dir, "recon_samples.npy"), xh[:k].cpu().numpy())
            saved = True

    all_mse = torch.cat(all_mse).numpy(); all_r2 = torch.cat(all_r2).numpy()
    summary = {"mode":"direct","mean_mse":float(all_mse.mean()),"std_mse":float(all_mse.std()),
               "mean_r2":float(all_r2.mean()),"std_r2":float(all_r2.std()),"N":int(all_mse.shape[0])}
    with open(os.path.join(args.save_dir, "recon_summary.json"), "w") as f: json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


@torch.no_grad()
def eval_from_codes(args):
    assert args.codes_dir is not None, "--codes_dir required for from_codes mode"
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths = [s.strip() for s in args.paths.split(",")]
    datasets = [VisualNSDPatched(p, patch_size=args.patch_len, zscore=args.zscore) for p in paths]
    dataset = torch.utils.data.ConcatDataset(datasets)
    feature_dim = datasets[0].f
    meta = make_meta(feature_dim, args.patch_len)

    # Build decoder + codebook
    full = ViT_VQ_AE_1D(
        feature_dim=feature_dim, patch_len=args.patch_len, embed_dim=args.embed_dim,
        depth_enc=args.depth_enc, depth_dec=args.depth_dec, num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio, codebook_size=args.codebook_size, vq_beta=args.vq_beta
    ).to(device)
    full.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    decoder = full.decoder.to(device).eval()
    codebook = full.vq.codebook.to(device).eval()

    all_mse, all_r2 = [], []
    saved = False
    cursor = 0
    shards = list_code_shards(args.codes_dir)
    if args.shard_limit is not None:
        shards = shards[: args.shard_limit]

    for p in shards:
        codes = np.load(p)  # [T, N]
        T, Np = codes.shape
        codes_t = torch.from_numpy(codes).long().to(device)
        z_q = codebook(codes_t)              # [T, N, D]
        x_hat = decoder(z_q, meta)           # [T, F]

        # Align GT slice
        end = min(cursor + T, len(dataset))
        gt = torch.stack([dataset[i] for i in range(cursor, end)], dim=0).view(end - cursor, -1).to(device)
        cursor = end

        all_mse.append(mse_per_sample(x_hat[:gt.size(0)], gt).cpu())
        all_r2.append(r2_per_sample(x_hat[:gt.size(0)], gt).cpu())

        if not saved and args.save_samples > 0:
            k = min(args.save_samples, gt.size(0))
            np.save(os.path.join(args.save_dir, "orig_samples.npy"), gt[:k].cpu().numpy())
            np.save(os.path.join(args.save_dir, "recon_samples.npy"), x_hat[:k].cpu().numpy())
            saved = True

        if cursor >= len(dataset):
            break

    all_mse = torch.cat(all_mse).numpy(); all_r2 = torch.cat(all_r2).numpy()
    summary = {"mode":"from_codes","mean_mse":float(all_mse.mean()),"std_mse":float(all_mse.std()),
               "mean_r2":float(all_r2.mean()),"std_r2":float(all_r2.std()),"N":int(all_mse.shape[0])}
    with open(os.path.join(args.save_dir, "recon_summary.json"), "w") as f: json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def main():
    args = parse_args()
    if args.mode == "direct":
        eval_direct(args)
    else:
        eval_from_codes(args)

if __name__ == "__main__":
    main()
