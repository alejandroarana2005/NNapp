# INTEGRANTES: ALEJANDRO ARANA FERNADEZ 2220232039

KGAT_cf4c965f90728dd9f994009ed1c855cf
# Install the viraual environment
> python -m venv .venv
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

# with poetr
Pretty soon

