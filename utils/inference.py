# utils/inference.py
import numpy as np
import logging
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)

ANIMAL_CLASSES = [
    "Bear", "Bird", "Cat", "Cow", "Deer",
    "Dog", "Dolphin", "Elephant", "Giraffe", "Horse",
    "Kangaroo", "Lion", "Panda", "Tiger", "Zebra"
]

def run_inference(x, framework, data_type, model_name="cnn"):
    if framework == "tensorflow":
        from models.tensorflow_models import get_model
        model = get_model(data_type)
        preds = model.predict(x)
    else:
        from models.pytorch_models import get_model
        # CAMBIO: pasar model_name para elegir CNN o ResNet
        model = get_model(data_type, model_name=model_name)
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32)
            logits   = model(x_tensor)
            preds    = F.softmax(logits, dim=1).cpu().numpy()

    if data_type == "tabular":
        if preds.ndim == 1 or preds.shape[1] == 1:
            classes = (preds > 0.5).astype(int).flatten()
        else:
            classes = np.argmax(preds, axis=1)
        return classes.tolist()

    elif data_type == "image":
        class_idx  = int(np.argmax(preds))
        confidence = float(np.max(preds)) * 100
        return {
            "animal":      ANIMAL_CLASSES[class_idx],
            "confidence":  f"{confidence:.1f}%",
            "model_used":  "ResNet18" if model_name == "resnet" else "CNN",
            "class_index": class_idx
        }

    return preds.tolist()