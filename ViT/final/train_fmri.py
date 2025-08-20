import os
import sys
import torch
import torch.nn as nn
import yaml
from tqdm import trange
from logger import Logger
from autoencoder1d import Autoencoder1D
from quantizer_ema import VectorQuantizer
from fmri_numpy import make_multi_loader

def load_config(cfg_path: str):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def get_temp(epoch, cfg):
    w = cfg["vq"]["warmup_epochs"]
    s = cfg["vq"]["soft_epochs"]
    if epoch < w:
        return 1.0
    elif epoch < w + s:
        t0, t1 = cfg["vq"]["temp_start"], cfg["vq"]["temp_end"]
        r = (epoch - w) / s
        return t0 + (t1 - t0) * r
    else:
        return 0.0

def main(cfg_path):
    cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = Logger(cfg["logging"]["out_dir"])

    # === Data ===
    train_set, dl_train = make_multi_loader(cfg["data"]["train"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        patch_size=cfg["model"]["embedding_dim"],
        zscore=cfg.get("zscore", True)
    )
    val_set, dl_val = make_multi_loader(cfg["data"]["val"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        patch_size=cfg["model"]["embedding_dim"],
        zscore=cfg.get("zscore", True)
    )

    # === Model & Quantizer ===
    model = Autoencoder1D(cfg["model"]).to(device)
    quantizer = VectorQuantizerEMA(
        n_codes=cfg["model"]["codebook_size"],
        embedding_dim=cfg["model"]["embedding_dim"],
        decay=cfg["vq"]["ema_decay"],
        entropy_lambda=cfg["vq"]["entropy_lambda"],
        use_cosine=cfg["vq"]["use_cosine"]
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"])
    scaler = torch.cuda.amp.GradScaler()
    loss_fn = nn.MSELoss()

    best_mse = float("inf")

    for epoch in trange(cfg["optim"]["epochs"], desc="Epochs"):
        model.train()
        total_loss, total_recon, total_vq = 0, 0, 0

        for xb in dl_train:
            xb = xb.to(device)
            optim.zero_grad()

            with torch.cuda.amp.autocast():
                z_e = model.encode(xb)
                temp = get_temp(epoch, cfg)
                z_q_st, vq_loss, _ = quantizer(z_e, beta_commit=cfg["vq"]["beta_commit"], temp=temp)
                xhat = model.decode(z_q_st)
                recon_loss = loss_fn(xhat, xb)
                loss = recon_loss + vq_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optim"]["grad_clip"])
            scaler.step(optim)
            scaler.update()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_vq += vq_loss.item()

        # === Evaluation ===
        model.eval()
        val_loss, val_mse = 0, 0
        last_info = {}
        with torch.no_grad():
            for xb in dl_val:
                xb = xb.to(device)
                z_e = model.encode(xb)
                z_q_st, _, info = quantizer(z_e, temp=0.0)
                xhat = model.decode(z_q_st)
                loss = loss_fn(xhat, xb)
                val_loss += loss.item()
                val_mse += nn.functional.mse_loss(xhat, xb).item()
                last_info = info  # last batch

        # === Logging ===
        logger.log({
            "epoch": epoch,
            "train/loss": total_loss / len(dl_train),
            "train/mse": total_recon / len(dl_train),
            "train/vq": total_vq / len(dl_train),
            "val/loss": val_loss / len(dl_val),
            "val/mse": val_mse / len(dl_val),
            "val/used_codes": last_info.get("used_codes", 0),
            "val/perplexity": last_info.get("perplexity", 0.0),
            "val/entropy": last_info.get("entropy", 0.0),
        })

        # === Checkpoint ===
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "quantizer": quantizer.state_dict()
        }
        ckpt_dir = os.path.join(cfg["logging"]["out_dir"], "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        logger.save_ckpt(os.path.join(ckpt_dir, f"epoch{epoch:03}.pt"), state)

        if val_mse < best_mse:
            best_mse = val_mse
            logger.save_ckpt(os.path.join(cfg["logging"]["out_dir"], "best.pt"), state)

        # === Dead code revive ===
        if (cfg["vq"]["dead_check_every"] > 0) and ((epoch + 1) % cfg["vq"]["dead_check_every"] == 0):
            total_tokens = len(train_set) * cfg["model"]["seq_len"]
            threshold = total_tokens * cfg["vq"]["dead_code_thr_pct"]
            quantizer.revive_dead_codes(z_e.reshape(-1, z_e.shape[-1]), threshold=threshold)

if __name__ == "__main__":
    main("experiment2/vitvq/final/config.yaml")
