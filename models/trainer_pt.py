# app/models/trainer_pt.py
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .pytorch_arch import TabularNet
from pathlib import Path

def train_tabular(X_train, y_train, X_test, y_test, epochs=20):
    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)  # ← float32 correcto para BCELoss
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = TabularNet(input_dim=X_train.shape[1])  # ← 30 features automático
    loss_fn = nn.BCELoss()   # ← CAMBIO: BCELoss para clasificación binaria
    opt = optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb).squeeze()  # ← squeeze: [batch,1] → [batch]
            loss = loss_fn(pred, yb)    # ← pred y yb tienen misma forma
            loss.backward()
            opt.step()

    Path("models/saved").mkdir(exist_ok=True)
    save_path = Path("models/saved/pt_tabular.pt")
    torch.save(model.state_dict(), save_path)
    return save_path