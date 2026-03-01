@echo off
setlocal enabledelayedexpansion
echo.
echo ==========================================
echo  Smart Surveillance Sorter - Installer
echo  Windows
echo ==========================================
echo.

:: Gestione parametri
set MODE=auto
for %%a in (%*) do (
    if "%%a"=="--use-rocm" set MODE=rocm
    if "%%a"=="--use-cuda" set MODE=cuda
    if "%%a"=="--use-cpu"  set MODE=cpu
)

:: 0. Abilita esecuzione script PowerShell (necessario per attivare la venv)
echo Configurazione PowerShell...
powershell -Command "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force" >nul 2>&1

:: 1. Cerca Python 3.12
echo Ricerca Python 3.12...
set PYTHON_BIN=
for %%p in (
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "C:\Python312\python.exe"
    "C:\Program Files\Python312\python.exe"
) do (
    if exist "%%~p" (
        set PYTHON_BIN=%%~p
        goto :found_python
    )
)
:: Prova con py launcher
py -3.12 --version >nul 2>&1
if %errorlevel% == 0 (
    set PYTHON_BIN=py -3.12
    goto :found_python
)
echo.
echo ERRORE: Python 3.12 non trovato.
echo Scaricalo da: https://www.python.org/downloads/release/python-31212/
echo Durante l'installazione seleziona "Add Python to PATH"
echo.
pause
exit /b 1

:found_python
echo Trovato: %PYTHON_BIN%
echo.

:: 2. Crea venv
if exist .venv (
    echo Rimozione venv esistente...
    rmdir /s /q .venv
)
echo Creazione venv...
"%PYTHON_BIN%" -m venv .venv
if %errorlevel% neq 0 (
    py -3.12 -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet
echo.

:: 3. PyTorch
if "%MODE%"=="rocm" goto :install_rocm
if "%MODE%"=="cuda" goto :install_cuda
if "%MODE%"=="cpu"  goto :install_cpu

:: Auto-detect NVIDIA
nvidia-smi >nul 2>&1
if %errorlevel% == 0 goto :install_cuda

:: Auto-detect AMD ROCm (controlla se rocminfo esiste)
where rocminfo >nul 2>&1
if %errorlevel% == 0 goto :install_rocm

:: Default CPU
goto :install_cpu

:install_rocm
echo GPU AMD rilevata - installazione PyTorch ROCm 7.2 per Windows...
echo Richiede driver AMD Adrenalin 26.1.1 o superiore.
echo.
echo Step 1/2: ROCm SDK (download pesante, attendere)...
pip install --no-cache-dir ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_core-7.2.0.dev0-py3-none-win_amd64.whl ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_devel-7.2.0.dev0-py3-none-win_amd64.whl ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_libraries_custom-7.2.0.dev0-py3-none-win_amd64.whl ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm-7.2.0.dev0.tar.gz
echo Step 2/2: PyTorch...
pip install --no-cache-dir ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torch-2.9.1+rocmsdk20260116-cp312-cp312-win_amd64.whl ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torchaudio-2.9.1+rocmsdk20260116-cp312-cp312-win_amd64.whl ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torchvision-0.24.1+rocmsdk20260116-cp312-cp312-win_amd64.whl
goto :install_deps

:install_cuda
echo GPU NVIDIA rilevata - installazione PyTorch CUDA...
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 ^
    --extra-index-url https://download.pytorch.org/whl/cu124 --quiet
goto :install_deps

:install_cpu
echo Nessuna GPU rilevata - installazione PyTorch CPU...
echo Per AMD: installa driver Adrenalin 26.1.1+ e rilancia con --use-rocm
echo Per NVIDIA: rilancia con --use-cuda
pip install torch torchvision torchaudio --quiet
goto :install_deps

:install_deps
echo.
echo Installazione dipendenze...
:: Filtra torch/torchvision/torchaudio dal requirements.txt
python -c "
lines = open('requirements.txt').readlines()
filtered = [l for l in lines if not l.lower().startswith(('torch','torchvision','torchaudio','pytorch-triton','rocm'))]
open('_req_tmp.txt', 'w').writelines(filtered)
"
pip install -r _req_tmp.txt --quiet
del _req_tmp.txt

:: Crea cartelle necessarie
if not exist models mkdir models
if not exist logs mkdir logs

:: Installa pacchetto
echo Installazione pacchetto...
pip install -e . --quiet

:: Verifica
echo.
echo Verifica installazione...
python -c "
import sys, torch
print(f'Python  : {sys.version.split()[0]}')
print(f'PyTorch : {torch.__version__}')
if torch.cuda.is_available():
    print(f'GPU     : {torch.cuda.get_device_name(0)}')
    try:
        x = torch.rand(1, device='cuda')
        print('VRAM    : OK')
    except Exception as e:
        print(f'VRAM    : FAILED ({e})')
else:
    print('GPU     : Non rilevata (CPU mode)')
    print('         Per AMD: installa driver Adrenalin 26.1.1+ e rilancia con --use-rocm')
"

echo.
echo Installazione completata!
echo Avvia con: run.bat
echo.
pause
