#!/bin/bash

# Uscire in caso di errore
set -e

echo "🚀 AI Surveillance Sorter - AMD/ROCm Installer"

# 1. Controllo versione Python 3.12
PYTHON_BIN=$(which python3.12)
if [ -z "$PYTHON_BIN" ]; then
    echo "❌ Errore: Python 3.12 non trovato. Installalo con: sudo apt install python3.12 python3.12-venv"
    exit 1
fi

# 2. Creazione VENV pulita
if [ -d ".venv" ]; then
    echo "🧹 Pulizia venv esistente..."
    rm -rf .venv
fi

$PYTHON_BIN -m venv .venv
source .venv/bin/activate

# 2. Rilevamento Hardware
echo "🔍 Rilevamento Hardware..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ Rilevata GPU NVIDIA (CUDA)"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
elif command -v rocminfo &> /dev/null; then
    # 4. Installazione specifica per AMD ROCm 6.4
    echo "⚙️ Installazione Torch 2.9.1 + ROCm 6.4"
    pip install torch==2.9.1+rocm6.4 torchvision==0.24.1+rocm6.4 torchaudio==2.9.1+rocm6.4 \
    --index-url https://download.pytorch.org/whl/rocm6.4
else
    echo "ℹ️ Nessuna GPU supportata trovata. Uso CPU."
    pip install torch torchvision
fi

# 5. Installazione dipendenze progetto
echo "📦 Installazione dipendenze comuni..."
pip install ultralytics ollama streamlit opencv-python pillow PyYAML

# 6. Verifica finale
echo "🔍 Verifica accelerazione hardware..."
python3 << END
import torch
print(f"Versione Torch: {torch.__version__}")
if torch.cuda.is_available():
    print("✅ ROCm/HIP rilevato correttamente!")
    print(f"Device: {torch.cuda.get_device_name(0)}")
else:
    print("❌ ATTENZIONE: Accelerazione hardware non rilevata. Verificare i driver ROCm del sistema.")
END

echo "✅ Installazione completata!"