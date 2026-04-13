"""
Funciones de preprocesamiento para datos tabulares, de imagen y de audio.
Se usan tanto en entrenamiento como en predicción.
"""
import io
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import StandardScaler

# Opcionales: solo si planeas usar imágenes/audio
from PIL import Image
import librosa


# -----------------------------
# Tabular
# -----------------------------
def preprocess_tabular(df: pd.DataFrame, target_column: str = None) -> tuple[np.ndarray, np.ndarray | None]:
    df = df.copy()

    # Separar y antes de limpiar X
    y = None
    if target_column and target_column in df.columns:
        y = df[target_column].values
        df = df.drop(columns=[target_column])

    # Limpiar columnas con strings numéricos mal formateados
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = (
                df[col]
                .str.replace('.', '', regex=False)  # quitar puntos de miles
                .str.replace(',', '.', regex=False) # coma decimal a punto
            )
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.fillna(0)

    scaler = StandardScaler()
    X = scaler.fit_transform(df.astype(np.float32))

    return X, y
# -----------------------------
def preprocess_image(file_bytes: bytes, framework: str = "pytorch", model_name: str = "cnn") -> np.ndarray:
    # CAMBIO: tamaño según modelo
    size = (224, 224) if model_name == "resnet" else (128, 128)

    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(size)
    arr = np.array(img).astype("float32") / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    arr  = (arr - mean) / std

    if framework == "pytorch":
        arr = arr.transpose(2, 0, 1)
        arr = arr[np.newaxis, ...]
    else:
        arr = arr[np.newaxis, ...]

    return arr.astype("float32")

# -----------------------------
# Audio
# -----------------------------
def preprocess_audio(file_bytes: bytes, sr: int = 16000) -> np.ndarray:
    """
    Carga un archivo de audio en memoria, lo resamplea y extrae un mel-spectrogram.
    Retorna un array 2D (frecuencias x frames).
    """
    # librosa.load acepta un file-like object
    audio, _ = librosa.load(io.BytesIO(file_bytes), sr=sr, mono=True)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Normalizamos a [0,1]
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_db.astype("float32")
