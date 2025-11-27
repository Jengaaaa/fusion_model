import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import AVECDataset
from fusion import FusionNet


# -------------------------------------------------------
# Load device
# -------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


# -------------------------------------------------------
# Load model
# -------------------------------------------------------
model = FusionNet().to(device)
model.load_state_dict(torch.load("checkpoints/best_fusionnet.pth", map_location=device))
model.eval()
print("✔ Loaded best_fusionnet.pth")


# -------------------------------------------------------
# Load Testing dataset
# -------------------------------------------------------
BASE_PATH = "/home/Pdanova/Dataset/Dataset/AVEC2014_AudioVisual"

test_dataset = AVECDataset(BASE_PATH, mode="Testing", tasks=["Freeform", "Northwind"])
loader_test = DataLoader(test_dataset, batch_size=1, shuffle=False)

print(f"Test samples: {len(test_dataset)}")


# -------------------------------------------------------
# Predict (multiple MFCC chunks)
# -------------------------------------------------------
def predict_sample(frames, mfcc_chunks):
    frames = frames.to(device)
    preds = []

    with torch.no_grad():
        for mfcc in mfcc_chunks:
            mfcc = mfcc.unsqueeze(0).to(device)
            out = model(frames, mfcc)
            preds.append(out.item())

    return sum(preds) / len(preds)


# -------------------------------------------------------
# Run inference
# -------------------------------------------------------
results = []

for frames, mfcc_chunks, prefix in tqdm(loader_test, desc="[Predict] Testing"):

    mfcc_chunks = mfcc_chunks[0]   # unwrap list

    pred = predict_sample(frames, mfcc_chunks)

    results.append({
        "prefix": prefix[0],
        "prediction": pred
    })


# -------------------------------------------------------
# Save CSV
# -------------------------------------------------------
df = pd.DataFrame(results)
df.to_csv("predictions_test.csv", index=False)

print("✔ Saved predictions_test.csv")
