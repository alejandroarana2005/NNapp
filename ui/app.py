import streamlit as st
import requests
import pandas as pd
from torch import nn
from torch import nn
from torchvision import models

FASTAPI_URL = "http://localhost:8000"

st.title("Demo Microservicio / PyTorch")

data_type = st.selectbox("Tipo de dato", ["tabular", "image", "audio"])
framework = st.selectbox("Framework", ["tensorflow", "pytorch"])
mode = st.radio("Modo de operación", ["Entrenar", "Predecir"])   

if mode == "Entrenar" and data_type == "tabular":
    st.subheader("Entrenamiento")
    csv_file = st.file_uploader("Sube CSV de entrenamiento", type="csv")
    if csv_file is not None:
        # Read the uploaded CSV into a DataFrame
        df = pd.read_csv(csv_file)
        st.subheader("First 5 Rows")
        st.dataframe(df.head())   # shows only the first 5 rows
        # Dropdown con las columnas disponibles para seleccionar el target
        target_column = st.selectbox("Selecciona la columna target", options=df.columns.tolist())
    else:
        target_column = None

    epochs = st.slider("Épocas", 1, 100, 20)
    if csv_file and target_column and st.button("Entrenar"):
        files = {"csv_file": (csv_file.name, csv_file.getvalue(), "text/csv")}
        data = {"framework": framework, 
                "data_type": data_type, 
                "epochs": str(epochs), 
                "target_column": target_column}
        with st.spinner("Entrenando el modelo..."):
            res = requests.post(f"{FASTAPI_URL}/train/", files=files, data=data)
        st.write("Status:", res.status_code)
        st.write("Raw response:", res.text)

        st.write(res.json())

elif mode == "Predecir":
    if data_type == "tabular":
        st.subheader("Predicción")
        csv_file = st.file_uploader("Sube CSV a predecir", type="csv")
        if csv_file is not None:
            # Read the uploaded CSV into a DataFrame
            df = pd.read_csv(csv_file)

            st.subheader("First 5 Rows")
            st.dataframe(df.head())   # shows only the first 5 rows
        
        if csv_file and st.button("Predecir"):
            # OJO: para FastAPI mandamos multipart/form-data con 'files'
            files = {
                "csv_file": (csv_file.name, csv_file.getvalue(), "text/csv")
            }
            data = {
                "data_type": "tabular",
                "framework": framework,  # el que hayas seleccionado en tu UI
            }

            # Ahora sí: enviar archivo + form-data a FastAPI
            res = requests.post(f"{FASTAPI_URL}/predict/", data=data, files=files)

            st.write("Status:", res.status_code)
            st.write("Raw response:", res.text)

            if res.ok:
                st.success("Predicción:")
                st.json(res.json())
            else:
                st.error("Error en la predicción")
    else:
        if data_type == "image" and framework == "pytorch":
            model_name = st.selectbox(
                "Modelo",
                ["cnn", "resnet"],
                format_func=lambda x: "CNN simple" if x == "cnn" else "ResNet18 (Transfer Learning)"
            )
        else:
            model_name = "cnn"

        file = st.file_uploader("Sube imagen", type=["png", "jpg", "jpeg"])
        if file and st.button("Predecir"):
            data  = {
                "data_type":  data_type,
                "framework":  framework,
                "model_name": model_name   # ← NUEVO
            }
            files = {"file": (file.name, file.getvalue())}
            r = requests.post(f"{FASTAPI_URL}/predict/", data=data, files=files)
            st.write("Status:", r.status_code)
            if r.ok:
                st.success("Predicción:")
                st.json(r.json())
            else:
                st.error("Error en la predicción")
                st.write(r.text)
            
            
class ResNetTransfer(nn.Module):
    def __init__(self, n_classes=15):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Congelar capas preentrenadas
        for param in self.model.parameters():
            param.requires_grad = False
        # Reemplazar capa final
        self.model.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        return self.model(x)
