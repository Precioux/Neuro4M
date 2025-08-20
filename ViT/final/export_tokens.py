import os
import numpy as np
import torch
import yaml
from tqdm import tqdm

from vit_vqae import ViTVQAE1D
from quantizer_ema import VectorQuantizerEMA
from fmri_numpy import make_multi_loader

def export(cfg_path):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(cfg["logging"]["out_dir"], "tokens")
    os.makedirs(out_dir, exist_ok=True)

    # --- Data ---
    patch_size = cfg["data"]["patch_size"]
    batch_size = cfg["data"]["batch_size"]
    num_workers = cfg["data"]["num_workers"]
    zscore = cfg["data"]["zscore"]

    loaders = {}
    for split in ["train", "val"]:
        npy_paths = cfg["data"][f"{split}_paths"]
        _, loader = make_multi_loader(npy_paths, batch_size, num_workers, patch_size, zscore)
        loaders[split] = loader

    # --- Build model ---
    dummy_x = next(iter(loaders["train"])).to(device)  # [B, L, P]
    seq_len = dummy_x.shape[1]
    code_dim = cfg["model"]["embedding_dim"]

    model = ViTVQAE1D(
        seq_len=seq_len,
        code_dim=code_dim,
        d_model=cfg['model']['vit']['dim'],
        depth=cfg['model']['vit']['depth'],
        heads=cfg['model']['vit']['heads'],
        mlp_ratio=cfg['model']['vit']['mlp_ratio'],
        dropout=cfg['model']['vit']['dropout'],
        token_dropout=0.0,
        pre_norm=cfg['model']['vit']['pre_norm']
    ).to(device)

    # Lazy init decoder
    model(dummy_x)
    model._lazy_set_dec_out(dummy_x)

    quantizer = VectorQuantizerEMA(
        n_codes=cfg["model"]["codebook_size"],
        embedding_dim=code_dim,
        decay=cfg["vq"]["ema_decay"],
        entropy_lambda=cfg["vq"]["entropy_lambda"],
        use_cosine=cfg["vq"]["use_cosine"]
    ).to(device)

    # --- Load best checkpoint ---
    ckpt_path = os.path.join(cfg["logging"]["out_dir"], "best.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    quantizer.load_state_dict(ckpt["quantizer"])
    model.eval()

    for split, loader in loaders.items():
        all_ze, all_zq, all_xhat, all_x = [], [], [], []

        for x in tqdm(loader, desc=f"Exporting {split}"):
            x = x.to(device)
            with torch.no_grad():
                z_e = model.encode(x)
                z_q, _, _ = quantizer(z_e, beta_commit=0.0, temp=0.0)
                x_hat = model.decode(z_q)

            all_ze.append(z_e.cpu().numpy())
            all_zq.append(z_q.cpu().numpy())
            all_xhat.append(x_hat.cpu().numpy())
            all_x.append(x.cpu().numpy())

        np.save(os.path.join(out_dir, f"{split}_z_e.npy"), np.concatenate(all_ze, axis=0))
        np.save(os.path.join(out_dir, f"{split}_z_q.npy"), np.concatenate(all_zq, axis=0))
        np.save(os.path.join(out_dir, f"{split}_xhat.npy"), np.concatenate(all_xhat, axis=0))
        np.save(os.path.join(out_dir, f"{split}_x.npy"), np.concatenate(all_x, axis=0))

        print(f"[✓] Saved {split} outputs to: {out_dir}/")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg", type=str)
    args = ap.parse_args()
    export(args.cfg)
