#!/bin/bash
APP_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$APP_PATH"

source .venv/bin/activate
# Check updates (only if git is available)
CURRENT=$(git rev-parse HEAD 2>/dev/null)
REMOTE=$(git ls-remote origin HEAD 2>/dev/null | cut -f1)
if [ "$CURRENT" != "$REMOTE" ] && [ -n "$REMOTE" ]; then
    echo "New version! Do: git pull && pip install -e ."
fi
# Export the PYTHONPATH to allow imports from the src package
export PYTHONPATH="$APP_PATH/src"

echo "Start Gradio WebUI - Smart Surveillance Sorter..."

python src/smart_surveillance_sorter/webui.py