# INTEGRANTES: ALEJANDRO ARANA FERNADEZ 2220232039

# Install the viraual environment
> python -m venv .venv
# versión compatible con tensor
> py -3.12 -m venv .venv 

# Activate the venv in Windows
> .\.venv\Scripts\activate

# Activate the venv in Linux
> source .venv\bin\activate

# install requirements packages
> pip install -r .\requirements.txt

# Abre un terminal y ejecutas el Backend:
> uvicorn app.main:app --reload --port 8000

# abre otro terminal y ejecutas el frontned
> streamlit run ui/app.py

# notas
Los entrenamientos se hicieron a través de los archivos localizados en NNapp\Scripts

# resultados

CNN 32X32  ------- 64% de precisión
CNN 124x124 ------------ 78% de precision
Transfer Learning ResNet 224x224 ---------------- 95.4% de precisión