@echo off
set APP_PATH=%~dp0
cd /d %APP_PATH%

call .venv\Scripts\activate.bat

set PYTHONPATH=%APP_PATH%src

echo 🌟 Avvio Gradio WebUI per Smart Surveillance Sorter...

python src\smart_surveillance_sorter\webui.py
pause
