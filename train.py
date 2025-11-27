import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from scipy.stats import pearsonr

from dataset import AVECDataset
from fusion import FusionNet


# ============================================================
# ✔ Pearson Loss (1 - r)
# ============================================================
def pearson_loss(pred, target):
    # pred, target: (B,)
    pred = pred - pred.mean()
    target = target - target.mean()
    
    num = (pred * target).sum()
    denom = torch.sqrt((pred ** 2).sum()) * torch.sqrt((target ** 2).sum())
    denom = denom + 1e-8  # 0 나누기 방지

    return 1 - num / denom   # 값이 작을수록 좋음 (r이 클수록)


# ============================================================
# ✔ Device
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ============================================================
# ✔ Dataset & Split
# ============================================================
BASE_PATH = "/home/Pdanova/Dataset/Dataset/AVEC2014_AudioVisual"

# Training 전체(Freeform + Northwind)를 하나의 dataset으로 사용
full_dataset = AVECDataset(BASE_PATH, mode="Training", tasks=["Freeform", "Northwind"])

# 80% train / 20% validation
train_size = int(len(full_dataset) * 0.8)
val_size = len(full_dataset) - train_size
train_set, val_set = random_split(full_dataset, [train_size, val_size])

print(f"Train size: {train_size}, Val size: {val_size}")

train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_set,   batch_size=4, shuffle=False, num_workers=0)


# ============================================================
# ✔ Model / Optim / Mixed Precision
# ============================================================
model = FusionNet().to(device)

mse_loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5)

scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

EPOCHS = 20
best_val_rmse = float("inf")

os.makedirs("checkpoints", exist_ok=True)
ckpt_path = "checkpoints/best_fusionnet.pth"


# ============================================================
# ✔ Training Loop
# ============================================================
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_losses = []
    train_preds, train_gts = [], []

    loop = tqdm(train_loader, desc=f"[Train] Epoch {epoch}/{EPOCHS}")

    for frames, mfcc, labels in loop:
        frames = frames.to(device)
        mfcc   = mfcc.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            out = model(frames, mfcc).squeeze()   # (B)
            loss_mse = mse_loss(out, labels)
            loss_p = pearson_loss(out, labels)
            loss = loss_mse + 0.2 * loss_p        # 가중치는 필요에 따라 조정

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_losses.append(loss.item())
        train_preds.extend(out.detach().cpu().numpy())
        train_gts.extend(labels.detach().cpu().numpy())

        loop.set_postfix(loss=loss.item())

    # --- Train metrics ---
    train_preds = np.array(train_preds)
    train_gts = np.array(train_gts)
    train_mae = np.mean(np.abs(train_preds - train_gts))
    train_rmse = np.sqrt(np.mean((train_preds - train_gts) ** 2))
    train_corr = pearsonr(train_preds, train_gts)[0]

    # ========================================================
    # ✔ Validation
    # ========================================================
    model.eval()
    val_preds, val_gts = [], []

    with torch.no_grad():
        for frames, mfcc, labels in val_loader:
            frames = frames.to(device)
            mfcc   = mfcc.to(device)
            labels = labels.to(device)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                out = model(frames, mfcc).squeeze()

            val_preds.extend(out.detach().cpu().numpy())
            val_gts.extend(labels.detach().cpu().numpy())

    val_preds = np.array(val_preds)
    val_gts = np.array(val_gts)
    val_mae = np.mean(np.abs(val_preds - val_gts))
    val_rmse = np.sqrt(np.mean((val_preds - val_gts) ** 2))
    val_corr = pearsonr(val_preds, val_gts)[0]

    print(
        f"\n[Epoch {epoch}] "
        f"Train Loss={np.mean(train_losses):.4f} | "
        f"Train MAE={train_mae:.3f} RMSE={train_rmse:.3f} Corr={train_corr:.3f} || "
        f"Val MAE={val_mae:.3f} RMSE={val_rmse:.3f} Corr={val_corr:.3f}"
    )

    # ========================================================
    # ✔ Best Model 저장 (Val RMSE 기준)
    # ========================================================
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        torch.save(model.state_dict(), ckpt_path)
        print(f"✅ Best model updated! (Val RMSE={best_val_rmse:.3f})\n")
