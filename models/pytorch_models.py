# models/pytorch_models.py
import torch
import torch.nn as nn
from torchvision import models

N_CLASSES = 15
_models   = {}

def get_model(data_type, model_name="cnn"):
    key = f"{data_type}_{model_name}"

    if key not in _models:
        if data_type == "tabular":
            from .pytorch_arch import TabularNet
            model = TabularNet()
            model.load_state_dict(torch.load(
                "models/saved/pt_tabular.pt",
                map_location="cpu"
            ))

        elif data_type == "image" and model_name == "resnet":
            # Cargar ResNet18 igual que como se entrenó
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            for param in model.parameters():
                param.requires_grad = False
            model.fc = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, N_CLASSES)
            )
            model.load_state_dict(torch.load(
                "models/saved/pt_resnet_224.pt",
                map_location="cpu"
            ))

        elif data_type == "image":
            from .pytorch_arch import ImageCNN
            model = ImageCNN()
            model.load_state_dict(torch.load(
                "models/saved/pt_image_128.pt",
                map_location="cpu"
            ))

        else:
            from .pytorch_arch import AudioCNN
            model = AudioCNN()
            model.load_state_dict(torch.load(
                "models/saved/pt_audio.pt",
                map_location="cpu"
            ))

        model.eval()
        _models[key] = model

    return _models[key]