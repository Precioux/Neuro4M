import os
import numpy as np
import yaml
from sklearn.metrics import mean_squared_error, r2_score

def compare_predictions(x_true, x_pred, name="Model"):
    assert x_true.shape == x_pred.shape, f"Shape mismatch: {x_true.shape} vs {x_pred.shape}"
    mse = mean_squared_error(x_true, x_pred)
    r2 = r2_score(x_true, x_pred)
    print(f"\n=== {name} ===")
    print(f"MSE : {mse:.6f}")
    print(f"R²  : {r2:.6f}")
    return {"mse": mse, "r2": r2}

def main(cfg_path):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    token_dir = os.path.join(cfg["logging"]["out_dir"], "tokens")
    x_gt_path = os.path.join(token_dir, "val_x.npy")
    x_model_path = os.path.join(token_dir, "val_xhat.npy")
    x_baseline_path = os.path.join(token_dir, "val_xhat_baseline.npy")  # optional

    assert os.path.exists(x_gt_path), f"Missing: {x_gt_path}"
    assert os.path.exists(x_model_path), f"Missing: {x_model_path}"

    x_gt = np.load(x_gt_path)
    x_model = np.load(x_model_path)

    compare_predictions(x_gt, x_model, name="ViT-VQ-AE")

    if os.path.exists(x_baseline_path):
        x_baseline = np.load(x_baseline_path)
        compare_predictions(x_gt, x_baseline, name="Baseline")
    else:
        print("⚠️  No baseline found at:", x_baseline_path)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cfg", type=str)
    args = ap.parse_args()
    main(args.cfg)
