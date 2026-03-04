@echo off
echo START BATCH
echo Current dir: %cd%
pushd "%~dp0" || (echo pushd FAILED & exit /b 1)
echo After pushd, dir: %cd%

call ".venv\Scripts\activate.bat"
echo After activate.bat

:: Ottimizzazioni CPU e GPU
set OMP_NUM_THREADS=%NUMBER_OF_PROCESSORS%
set MKL_NUM_THREADS=%NUMBER_OF_PROCESSORS%
set OLLAMA_VULKAN=1


:: Check aggiornamenti
for /f %%i in ('git rev-parse HEAD 2^>nul') do set CURRENT=%%i
for /f %%i in ('git ls-remote origin HEAD 2^>nul') do set REMOTE=%%i
if not "%CURRENT%"=="%REMOTE%" (
    if not "%REMOTE%"=="" (
        echo ⚠️  Nuova versione disponibile! Esegui: git pull ^&^& pip install -e .
    )
)
if not exist models mkdir models
if not exist logs mkdir logs
echo Directories OK

echo Start Gradio WebUI for Smart Surveillance Sorter...
python -m smart_surveillance_sorter.webui
echo After python -m webui

pause