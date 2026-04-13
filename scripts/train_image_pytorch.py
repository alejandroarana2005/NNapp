# train_image_pytorch.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import sys

# configuración del entrenamiento
DATASET_PATH = r"C:\Users\alejo\.cache\kagglehub\datasets\likhon148\animal-data\versions\1\animal_data"
EPOCHS       = 30
BATCH_SIZE   = 32
IMG_SIZE     = 32
LR           = 1e-3
SAVE_PATH = Path("../models/saved/pt_image_32.pt")

#  Clases del dataset 
CLASSES = [
    "Bear", "Bird", "Cat", "Cow", "Deer",
    "Dog", "Dolphin", "Elephant", "Giraffe", "Horse",
    "Kangaroo", "Lion", "Panda", "Tiger", "Zebra"
]
N_CLASSES = len(CLASSES)  #15

# Transformaciones de imagen 
transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),       # aumentación de datos
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

#  se carga del dataset 
print("Cargando dataset...")
full_dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform_train)

print(f"Clases encontradas: {full_dataset.classes}")
print(f"Total imágenes:     {len(full_dataset)}")

# División 80% train 20% validación
train_size = int(0.8 * len(full_dataset))
val_size   = len(full_dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

print(f"Train: {train_size} imágenes | Validación: {val_size} imágenes")

#  Modelo 
sys.path.insert(0, "..")
from models.pytorch_arch import ImageCNN

model   = ImageCNN(n_classes=N_CLASSES)
loss_fn = nn.CrossEntropyLoss()          # multiclase
opt     = optim.Adam(model.parameters(), lr=LR)

#  Entrenamiento 
print("\nIniciando entrenamiento...\n")

for epoch in range(1, EPOCHS + 1):
    #  Train
    model.train()
    train_loss, train_correct = 0, 0

    for xb, yb in train_loader:
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
        train_loss    += loss.item()
        train_correct += (pred.argmax(1) == yb).sum().item()

    # Validación 
    model.eval()
    val_loss, val_correct = 0, 0

    with torch.no_grad():
        for xb, yb in val_loader:
            pred       = model(xb)
            val_loss  += loss_fn(pred, yb).item()
            val_correct += (pred.argmax(1) == yb).sum().item()

    train_acc = train_correct / train_size * 100
    val_acc   = val_correct   / val_size   * 100

    print(f"Epoch {epoch:>2}/{EPOCHS} | "
          f"Train loss: {train_loss/len(train_loader):.4f} acc: {train_acc:.1f}% | "
          f"Val loss: {val_loss/len(val_loader):.4f} acc: {val_acc:.1f}%")

# Guardar modelo 
SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), SAVE_PATH)
print(f"\nModelo guardado en: {SAVE_PATH}")