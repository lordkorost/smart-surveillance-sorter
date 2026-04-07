#!/bin/bash
set -e

MODE="auto"
for arg in "$@"; do
    case $arg in
        --use-rocm) MODE="rocm" ;;
        --use-cuda) MODE="cuda" ;;
        --use-cpu)  MODE="cpu"  ;;
    esac
done

echo "🚀 Smart Surveillance Sorter - Installer (Linux)"
echo "================================================="

# 1. Python 3.12
PYTHON_BIN=$(which python3.12 2>/dev/null) || {
    echo "❌ Python 3.12 not found. Install it with: sudo apt install python3.12 python3.12-venv"
    exit 1
}
echo "✅ Python found: $($PYTHON_BIN --version)"

# 2. Virtual environment
if [ -d ".venv" ]; then
    echo "⚠️  Existing .venv folder found — it will be recreated."
    rm -rf .venv
fi
$PYTHON_BIN -m venv .venv
source .venv/bin/activate
pip install --upgrade pip --quiet

# 3. PyTorch (must be installed BEFORE requirements.txt)
if [ "$MODE" == "cuda" ] || ([ "$MODE" == "auto" ] && command -v nvidia-smi &> /dev/null); then
    echo "✅ NVIDIA GPU detected — installing PyTorch CUDA..."
    pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 \
        --extra-index-url https://download.pytorch.org/whl/cu124 --quiet

elif [ "$MODE" == "rocm" ] || ([ "$MODE" == "auto" ] && command -v rocminfo &> /dev/null); then
    echo "✅ AMD GPU detected — installing PyTorch ROCm 6.4..."
    pip install torch==2.9.1+rocm6.4 torchvision==0.24.1+rocm6.4 torchaudio==2.9.1+rocm6.4 \
        --index-url https://download.pytorch.org/whl/rocm6.4 --quiet

else
    echo "ℹ️  No GPU detected — installing PyTorch CPU..."
    pip install torch torchvision torchaudio --quiet
fi

# 4. Dependencies from requirements.txt (excludes torch/torchvision/torchaudio already installed)
echo "📦 Installing dependencies..."
grep -v "^torch\|^torchvision\|^torchaudio\|^pytorch-triton" requirements.txt | pip install -r /dev/stdin --quiet

# 5. Install the package
pip install -e . --quiet

# 6. Verify installation
echo ""
echo "🔍 Verifying installation..."
python3 - << 'PYEOF'
import sys, torch
print(f"Python  : {sys.version.split()[0]}")
print(f"PyTorch : {torch.__version__}")
is_rocm = "rocm" in torch.__version__
if torch.cuda.is_available():
    device = torch.cuda.get_device_name(0)
    backend = "ROCm/HIP (AMD)" if is_rocm else "CUDA (NVIDIA)"
    print(f"Backend : {backend}")
    print(f"GPU     : {device}")
    try:
        x = torch.rand(1, device="cuda")
        print("VRAM    : ✅ OK")
    except Exception as e:
        print(f"VRAM    : ❌ FAILED ({e})")
else:
    print("GPU     : ❌ Not detected — CPU mode")
    if is_rocm:
        print("Tip: check ROCm drivers with 'lsmod | grep amdgpu'")
    else:
        print("Tip: check NVIDIA drivers with 'nvidia-smi'")
PYEOF

if [ -f "run.sh" ]; then
    chmod +x run.sh
fi

echo ""
echo "✅ Installation complete!"
echo "   Launch with: ./run.sh"