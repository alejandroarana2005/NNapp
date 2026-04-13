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


