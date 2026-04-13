# scripts/train_resnet_pytorch.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
import sys

# ── Configuración ──────────────────────────────────────────────
DATASET_PATH = r"C:\Users\alejo\.cache\kagglehub\datasets\likhon148\animal-data\versions\1\animal_data"
EPOCHS       = 30
BATCH_SIZE   = 32
IMG_SIZE     = 224
LR           = 1e-4        # lr más bajo para fine-tuning
SAVE_PATH = Path("../models/saved/pt_resnet_224.pt")

N_CLASSES = 15

# ── Transformaciones ───────────────────────────────────────────
# ResNet fue preentrenado con 224x224 pero acepta 32x32
transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Carga del dataset ──────────────────────────────────────────
print("Cargando dataset...")
full_dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform_train)

print(f"Clases encontradas: {full_dataset.classes}")
print(f"Total imágenes:     {len(full_dataset)}")

train_size = int(0.8 * len(full_dataset))
val_size   = len(full_dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

print(f"Train: {train_size} imágenes | Validación: {val_size} imágenes")

# ── Modelo ResNet18 preentrenado ───────────────────────────────
print("\nCargando ResNet18 preentrenado...")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Congelar todas las capas — no modificar pesos preentrenados
for param in model.parameters():
    param.requires_grad = False

# Reemplazar solo la capa final para nuestras 15 clases
# ResNet18 tiene 512 neuronas en la última capa
model.fc = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, N_CLASSES)   # ← 15 clases de animales
)

# Solo los parámetros de la capa final se entrenan
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Parámetros entrenables: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

loss_fn = nn.CrossEntropyLoss()
# Solo optimizar la capa final
opt = optim.Adam(model.fc.parameters(), lr=LR)

# ── Entrenamiento ──────────────────────────────────────────────
print("\nIniciando entrenamiento ResNet18...\n")

best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    train_loss, train_correct = 0, 0

    for xb, yb in train_loader:
        opt.zero_grad()
        pred  = model(xb)
        loss  = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        train_loss    += loss.item()
        train_correct += (pred.argmax(1) == yb).sum().item()

    # Validación
    model.eval()
    val_loss, val_correct = 0, 0

    with torch.no_grad():
        for xb, yb in val_loader:
            pred        = model(xb)
            val_loss   += loss_fn(pred, yb).item()
            val_correct += (pred.argmax(1) == yb).sum().item()

    train_acc = train_correct / train_size * 100
    val_acc   = val_correct   / val_size   * 100

    # Guardar el mejor modelo
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"Epoch {epoch:>2}/{EPOCHS} | "
              f"Train loss: {train_loss/len(train_loader):.4f} acc: {train_acc:.1f}% | "
              f"Val loss: {val_loss/len(val_loader):.4f} acc: {val_acc:.1f}% ← mejor modelo guardado")
    else:
        print(f"Epoch {epoch:>2}/{EPOCHS} | "
              f"Train loss: {train_loss/len(train_loader):.4f} acc: {train_acc:.1f}% | "
              f"Val loss: {val_loss/len(val_loader):.4f} acc: {val_acc:.1f}%")

print(f"\nMejor val accuracy: {best_val_acc:.1f}%")
print(f"Modelo guardado en: {SAVE_PATH}")