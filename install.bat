@echo off
setlocal enabledelayedexpansion
echo.
echo ==========================================
echo  Smart Surveillance Sorter - Installer
echo  Windows
echo ==========================================
echo.

:: Parse arguments
set MODE=auto
for %%a in (%*) do (
    if "%%a"=="--use-rocm" set MODE=rocm
    if "%%a"=="--use-cuda" set MODE=cuda
    if "%%a"=="--use-cpu"  set MODE=cpu
)

:: 0. Enable PowerShell script execution (required to activate venv)
echo Configuring PowerShell...
powershell -Command "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force" >nul 2>&1

:: 1. Find Python 3.12
echo Looking for Python 3.12...
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
:: Try py launcher
py -3.12 --version >nul 2>&1
if %errorlevel% == 0 (
    set PYTHON_BIN=py -3.12
    goto :found_python
)
echo.
echo ERROR: Python 3.12 not found.
echo Download from: https://www.python.org/downloads/release/python-31212/
echo During installation, select "Add Python to PATH"
echo.
pause
exit /b 1

:found_python
echo Found: %PYTHON_BIN%
echo.

:: 2. Create virtual environment
if exist .venv (
    echo Removing existing .venv...
    rmdir /s /q .venv
)
echo Creating virtual environment...
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

:: Auto-detect AMD ROCm
where rocminfo >nul 2>&1
if %errorlevel% == 0 goto :install_rocm

:: Default CPU
goto :install_cpu

:install_rocm
echo AMD GPU detected - installing PyTorch ROCm 7.2 for Windows...
echo Requires AMD Adrenalin driver 26.1.1 or later.
echo.
echo Step 1/2: ROCm SDK (large download, please wait)...
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
echo NVIDIA GPU detected - installing PyTorch CUDA...
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 ^
    --extra-index-url https://download.pytorch.org/whl/cu124 --quiet
goto :install_deps

:install_cpu
echo No GPU detected - installing PyTorch CPU (with MKL)...
echo For AMD: install Adrenalin driver 26.1.1+ and re-run with --use-rocm
echo For NVIDIA: re-run with --use-cuda
echo.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
goto :install_deps

:install_deps
echo.
echo Installing dependencies...
python filter_requirements.py
pip install -r _req_tmp.txt --quiet
del _req_tmp.txt

if not exist models mkdir models
if not exist logs mkdir logs

echo Installing package...
pip install -e . --quiet

echo.
echo Verifying installation...
python verify_install.py

echo.
echo Installation complete!
echo Launch with: .\run.bat
echo.
pause
