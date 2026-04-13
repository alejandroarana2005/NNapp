# train_image_tf.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

# Configuración 
DATASET_PATH = r"C:\Users\alejo\.cache\kagglehub\datasets\likhon148\animal-data\versions\1\animal_data"
EPOCHS       = 30
BATCH_SIZE   = 32
IMG_SIZE     = (128, 128)
SAVE_PATH = "../models/saved/tf_image.keras"

#  Generadores con aumentación 
def preprocess_input_imagenet(img):
    img = img / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    return (img - mean) / std

import numpy as np
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input_imagenet,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=10,
    zoom_range=0.1
)


train_data = train_gen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    subset="training"
)

val_data = train_gen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    subset="validation"
)

print(f"\nClases encontradas: {list(train_data.class_indices.keys())}")
print(f"Train: {train_data.samples} imágenes")
print(f"Val:   {val_data.samples} imágenes")

# Modelo 
import sys
sys.path.insert(0, "..")
from models.tensorflow_arch import build_image_model

n_classes = len(train_data.class_indices)
model     = build_image_model(num_classes=n_classes)
model.summary()

#  Entrenamiento 
print("\nIniciando entrenamiento TensorFlow...\n")
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data
)

# Guardar modelo
Path("models/saved").mkdir(parents=True, exist_ok=True)
model.save(SAVE_PATH)
print(f"\nModelo guardado en: {SAVE_PATH}")