#!/bin/bash
set -e

# Inizializza variabile scelta
MODE="auto"

# Gestione parametri
for arg in "$@"; do
    case $arg in
        --use-rocm) MODE="rocm"; shift ;;
        --use-cuda) MODE="cuda"; shift ;;
    esac
done

echo "🚀 AI Surveillance Sorter - Professional Installer"

# 1. Controllo Python 3.12 (come prima)
PYTHON_BIN=$(which python3.12) || { echo "❌ Python 3.12 non trovato"; exit 1; }

# 2. Setup VENV
[ -d ".venv" ] && rm -rf .venv
$PYTHON_BIN -m venv .venv
source .venv/bin/activate

# 3. Logica di Installazione differenziata
if [ "$MODE" == "cuda" ] || ([ "$MODE" == "auto" ] && command -v nvidia-smi &> /dev/null); then
    echo "✅ Installazione per NVIDIA (CUDA)..."
    pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --extra-index-url https://download.pytorch.org/whl/cu124

elif [ "$MODE" == "rocm" ] || ([ "$MODE" == "auto" ] && command -v rocminfo &> /dev/null); then
    echo "⚙️ Installazione per AMD (ROCm 6.4)..."
    pip install torch==2.9.1+rocm6.4 torchvision==0.24.1+rocm6.4 torchaudio==2.9.1+rocm6.4 \
    --index-url https://download.pytorch.org/whl/rocm6.4
else
    echo "ℹ️ Installazione Standard (CPU)..."
    pip install torch torchvision
fi

# 4. Resto delle dipendenze e Verifica (come nel tuo script)
echo "📦 Installazione dipendenze comuni..."
echo "📦 Installazione dipendenze comuni..."
pip install \
    ultralytics \
    opencv-python \
    transformers \
    pillow \
    timm \
    psutil \
    GPUtil \
    colorama \
    tqdm \
    ollama \
    gradio \
    requests \
    pyyaml


print(f"--- Info Sistema ---")
print(f"Versione Torch: {torch.__version__}")
print(f"Python: {sys.version.split()[0]}")

# Verifica se il backend è ROCm o CUDA
is_rocm = "rocm" in torch.__version__
backend_name = "ROCm/HIP (AMD)" if is_rocm else "CUDA (NVIDIA)"

if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f"\n✅ Accelerazione hardware RILEVATA!")
    print(f"Backend attivo: {backend_name}")
    print(f"GPU rilevata: {device_name}")
    
    # Test rapido di allocazione memoria
    try:
        x = torch.rand(1, device="cuda")
        print("✅ Test allocazione VRAM: SUCCESS")
    except Exception as e:
        print(f"❌ Test allocazione VRAM: FAILED ({e})")
else:
    print(f"\n❌ ATTENZIONE: Accelerazione hardware NON rilevata.")
    if is_rocm:
        print("Sugerimento: Controlla che i driver ROCm siano caricati ('lsmod | grep amdgpu')")
    else:
        print("Suggerimento: Controlla i driver NVIDIA e il toolkit CUDA ('nvidia-smi')")
END

echo -e "\n✅ Procedura completata!"