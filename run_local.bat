@echo off
cd /d "%~dp0"
python --version
python -m pip install -r requirements.txt
python train.py
python -m streamlit run app.py
