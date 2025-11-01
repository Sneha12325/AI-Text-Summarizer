@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
  echo Creating virtual environment...
  python -m venv .venv
)

call .venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip >pip_upgrade.log 2>&1

if exist requirements.txt (
  pip install -r requirements.txt >install.log 2>install_errors.log
) else (
  pip install waitress flask transformers redis >install.log 2>install_errors.log
)

echo Starting server...
start "" cmd /c ".venv\Scripts\waitress-serve --listen=0.0.0.0:5000 app:app"

timeout /t 2 /nobreak >nul
start "" "http://localhost:5000"

echo Logs: install.log install_errors.log pip_upgrade.log
exit /b 0