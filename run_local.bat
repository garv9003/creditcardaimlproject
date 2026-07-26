@echo off
cd /d "%~dp0"
echo Checking Python...
python --version

echo Installing dependencies...
python -m pip install -r requirements.txt

echo Training model...
python train.py

echo Starting Streamlit app...
start "Streamlit App" cmd /k "python -m streamlit run app.py --server.headless true --server.address 127.0.0.1 --server.port 8501"

echo Waiting a few seconds for the app to start...
timeout /t 8 /nobreak >nul
start "" http://127.0.0.1:8501

echo Opened http://127.0.0.1:8501 in your browser.
pause
