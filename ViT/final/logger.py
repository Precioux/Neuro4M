import os
import csv
import logging

class Logger:
    def __init__(self, out_dir: str, name: str = "train"):
        os.makedirs(out_dir, exist_ok=True)
        self.log_file = os.path.join(out_dir, f"{name}.log")
        self.csv_file = os.path.join(out_dir, "metrics.csv")
        self._init_logger(name)
        self._init_csv()

    def _init_logger(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        fh = logging.FileHandler(self.log_file)
        ch = logging.StreamHandler()
        fh.setLevel(logging.INFO)
        ch.setLevel(logging.INFO)

        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def _init_csv(self):
        with open(self.csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_mse", "vq_loss", "val_loss", "val_mse", "used_codes", "perplexity"])

    def log(self, stats: dict):
        # CSV logging
        with open(self.csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                stats.get("epoch"),
                stats.get("train/loss"),
                stats.get("train/mse"),
                stats.get("train/vq"),
                stats.get("val/loss"),
                stats.get("val/mse"),
                stats.get("val/used_codes"),
                stats.get("val/perplexity"),
            ])
        # Console/file logging
        self.logger.info(
            f"[epoch {stats['epoch']:03}] "
            f"train loss={stats['train/loss']:.4f} | "
            f"recon={stats['train/mse']:.4f} | "
            f"vq={stats['train/vq']:.4f} || "
            f"val mse={stats['val/mse']:.4f} | "
            f"codes={stats['val/used_codes']}/{stats['val/perplexity']:.1f}"
        )

    def save_ckpt(self, path: str, model_state: dict):
        torch.save(model_state, path)
        self.logger.info(f"[✓] Saved checkpoint to {path}")
