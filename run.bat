@echo off
echo START BATCH
echo Current dir: %cd%
pushd "%~dp0" || (echo pushd FAILED & exit /b 1)
echo After pushd, dir: %cd%

call ".venv\Scripts\activate.bat"
echo After activate.bat

if not exist models mkdir models
if not exist logs mkdir logs
echo Directories OK

echo Start Gradio WebUI for Smart Surveillance Sorter...
python -m smart_surveillance_sorter.webui
echo After python -m webui

pause