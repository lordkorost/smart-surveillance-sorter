#!/bin/bash
APP_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$APP_PATH"

source .venv/bin/activate

# Esporta il PYTHONPATH per permettere gli import dal pacchetto src
export PYTHONPATH="$APP_PATH/src"

echo "🌟 Avvio Gradio WebUI per Smart Surveillance Sorter..."

# Cambiato da 'streamlit run' a 'python'
python src/smart_surveillance_sorter/webui.py