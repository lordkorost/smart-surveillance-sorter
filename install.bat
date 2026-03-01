@echo off
setlocal enabledelayedexpansion
echo.
echo ==========================================
echo  Smart Surveillance Sorter - Installer
echo  Windows (CPU only)
echo ==========================================
echo.
echo NOTE: Su Windows e' supportata solo la modalita' CPU.
echo Per GPU AMD o NVIDIA installa PyTorch manualmente nella .venv.
echo Vedi: https://pytorch.org/get-started/locally/
echo.

:: 1. Cerca Python 3.12
set PYTHON_BIN=
for %%p in (
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "C:\Python312\python.exe"
    "C:\Program Files\Python312\python.exe"
) do (
    if exist %%p (
        set PYTHON_BIN=%%p
        goto :found_python
    )
)

:: Prova con py launcher
py -3.12 --version >nul 2>&1
if %errorlevel% == 0 (
    set PYTHON_BIN=py -3.12
    goto :found_python
)

echo ❌ Python 3.12 non trovato.
echo    Scaricalo da: https://www.python.org/downloads/release/python-3120/
echo    Assicurati di selezionare "Add Python to PATH" durante l'installazione.
pause
exit /b 1

:found_python
echo ✅ Python trovato: %PYTHON_BIN%
echo.

:: 2. Crea venv
if exist .venv (
    echo ⚠️  Cartella .venv esistente - verra' ricreata...
    rmdir /s /q .venv
)
%PYTHON_BIN% -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet

:: 3. PyTorch CPU
echo 📦 Installazione PyTorch (CPU)...
pip install torch torchvision torchaudio --quiet

:: 4. Dipendenze (esclude torch/torchvision/torchaudio)
echo 📦 Installazione dipendenze...
python -c "
lines = open('requirements.txt').readlines()
filtered = [l for l in lines if not l.startswith(('torch','torchvision','torchaudio','pytorch-triton'))]
open('_req_tmp.txt', 'w').writelines(filtered)
"
pip install -r _req_tmp.txt --quiet
del _req_tmp.txt

:: 5. Installa pacchetto
pip install -e . --quiet

:: 6. Verifica
echo.
echo 🔍 Verifica installazione...
python -c "
import sys, torch
print(f'Python  : {sys.version.split()[0]}')
print(f'PyTorch : {torch.__version__}')
if torch.cuda.is_available():
    print(f'GPU     : {torch.cuda.get_device_name(0)}')
else:
    print('GPU     : Non rilevata (CPU mode)')
"

echo.
echo ✅ Installazione completata!
echo    Avvia con: run.bat
echo.
pause
