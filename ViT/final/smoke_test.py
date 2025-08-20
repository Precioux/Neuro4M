import os, sys, subprocess, numpy as np

def main():
    os.makedirs("tmp_data", exist_ok=True)
    N, F = 2000, 2048
    np.save("tmp_data/fake_visual.npy", np.random.randn(N, F).astype("float32"))

    cmd = [
        sys.executable, "-m", "vitvq.scripts.train_then_export",
        "--paths", "tmp_data/fake_visual.npy",
        "--feature_dim", str(F),
        "--save_dir", "outputs/smoke_train",
        "--export_dir", "outputs/smoke_tokens",
        "--batch_size", "128",
        "--epochs", "2",
        "--lr", "3e-4",
        "--patch_len", "64",
        "--embed_dim", "128",
        "--codebook_size", "512",
        "--save_mse"
    ]
    print("Running:", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(rc)
    assert os.path.exists("outputs/smoke_train/best.pt")
    assert os.path.exists("outputs/smoke_tokens/codebook.npy")
    print("Smoke test passed.")

if __name__ == "__main__":
    main()
