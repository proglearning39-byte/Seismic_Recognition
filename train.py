import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from model import PatchTSTEEWRobust
from dataset import EEWRobustDataset

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using Device: {device}")

    # 模擬資料 (實作時請匯入真實 npy / mseed 資料)
    dummy_x = np.random.randn(200, 1600).astype(np.float32)
    dummy_y = np.random.randint(0, 2, size=(200,)).astype(np.float32)

    dataset = EEWRobustDataset(dummy_x, dummy_y)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = PatchTSTEEWRobust().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(5):
        total_loss = 0
        for x_b, y_b in dataloader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_b), y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), "patchtst_eew_best.pt")
    print("✅ Model saved to patchtst_eew_best.pt")

if __name__ == "__main__":
    train()
