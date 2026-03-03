#!/bin/bash
APP_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$APP_PATH"

source .venv/bin/activate
# Check aggiornamenti
CURRENT=$(git rev-parse HEAD 2>/dev/null)
REMOTE=$(git ls-remote origin HEAD 2>/dev/null | cut -f1)
if [ "$CURRENT" != "$REMOTE" ] && [ -n "$REMOTE" ]; then
    echo "⚠️  Nuova versione disponibile! Esegui: git pull && pip install -e ."
fi
# Esporta il PYTHONPATH per permettere gli import dal pacchetto src
export PYTHONPATH="$APP_PATH/src"

echo "🌟 Start Gradio WebUI - Smart Surveillance Sorter..."

python src/smart_surveillance_sorter/webui.py