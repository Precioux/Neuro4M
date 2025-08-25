import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import joblib
import os
import sys

# ========== Load Data ==========
print("📦 Loading data...", flush=True)
Z = np.load("/mnt/mahdipou/nsd/pca_results/pca_encoded.npy")  # Shape: [7500, 1500]
pca = joblib.load("/mnt/mahdipou/nsd/pca_results/pca_model.pkl")  # PCA model
print("✅ Data loaded.", flush=True)

# ========== Dataset & DataLoader ==========
Z_tensor = torch.tensor(Z, dtype=torch.float32)
dataset = TensorDataset(Z_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
print("🧾 Dataset created.", flush=True)

# ========== MLP Encoder ==========
class MLPEncoder(nn.Module):
    def __init__(self, input_dim=1500, latent_dim=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x):
        return self.model(x)

# ========== MLP Decoder ==========
class MLPDecoder(nn.Module):
    def __init__(self, latent_dim=64, output_dim=1500):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# ========== Initialize Models ==========
print("⚙️ Initializing models...", flush=True)
encoder = MLPEncoder()
decoder = MLPDecoder()

params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(params, lr=1e-3)
criterion = nn.MSELoss()

# ========== Training Loop ==========
print("🚀 Starting training...", flush=True)
num_epochs = 20
encoder.train()
decoder.train()

# log file
loss_log_path = "loss_log.txt"
with open(loss_log_path, "w") as f:
    f.write("epoch,loss\n")

for epoch in range(num_epochs):
    print(f"🟢 Epoch {epoch+1} started...", flush=True)
    total_loss = 0
    for batch in dataloader:
        x = batch[0]
        tokens = encoder(x)
        recon = decoder(tokens)
        loss = criterion(recon, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"✅ Epoch {epoch+1} completed. Loss: {avg_loss:.4f}", flush=True)

    # save loss log
    with open(loss_log_path, "a") as f:
        f.write(f"{epoch+1},{avg_loss:.6f}\n")

# ========== Encode & Decode All ==========
print("📤 Encoding and reconstructing full dataset...", flush=True)
encoder.eval()
decoder.eval()
with torch.no_grad():
    tokens = encoder(Z_tensor).numpy()
    Z_reconstructed = decoder(torch.tensor(tokens)).numpy()
    X_reconstructed = pca.inverse_transform(Z_reconstructed)

# ========== Save Outputs ==========
print("💾 Saving all outputs...", flush=True)
np.save("Z_tokens_mlp.npy", tokens)
np.save("Z_reconstructed_mlp.npy", Z_reconstructed)
np.save("X_reconstructed_mlp.npy", X_reconstructed)
torch.save(encoder.state_dict(), "mlp_encoder.pt")
torch.save(decoder.state_dict(), "mlp_decoder.pt")

print("🎉 Done! Everything saved successfully.", flush=True)
