# app/models/tensorflow_arch.py

from tensorflow import keras
from tensorflow.keras import layers

def build_tabular_model(input_dim=30):
  
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid")# 1 neurona para clasificación binaria (benigno/maligno)
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy", 
        metrics=["accuracy"]
    )
    return model

def build_image_model(num_classes=15): # 15 carpetas de animales diferentes
    model = keras.Sequential([
        layers.Input(shape=(128, 128, 3)),  # 128x128 RGB
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.GlobalAveragePooling2D(),   
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")  
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"]
    )
    return model

def build_audio_model(num_classes=35):
    model = keras.Sequential([
        layers.Conv2D(32, (3,3), activation="relu", input_shape=(None, None, 1)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3,3), activation="relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model