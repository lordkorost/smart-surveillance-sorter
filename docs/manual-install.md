# 🔧 Manual Installation

This guide is for users who prefer to set up the Python environment manually instead of using `install.sh` / `install.bat`.

> ℹ️ The automated installers (`install.sh` / `install.bat`) are the recommended way to install Smart Surveillance Sorter. Use this guide only if you have a specific reason to install manually.

---

## Prerequisites

- Python 3.12 installed
- For AMD GPU: ROCm drivers installed — see [AMD GPU Setup](gpu-setup-amd.md)
- For NVIDIA GPU: CUDA 12.x drivers installed

---

## 🐧 Linux

### AMD GPU (ROCm 6.4)

```bash
# Create venv
python3.12 -m venv .venv
source .venv/bin/activate

# Install PyTorch ROCm 6.4
pip install torch==2.9.1+rocm6.4 torchvision==0.24.1+rocm6.4 torchaudio==2.9.1+rocm6.4 \
    --index-url https://download.pytorch.org/whl/rocm6.4

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### NVIDIA GPU (CUDA)

> ⚠️ CUDA support has not been tested by the maintainers — no NVIDIA hardware was available for testing. The installation procedure follows standard PyTorch CUDA guidelines. Community feedback welcome.

```bash
# Create venv
python3.12 -m venv .venv
source .venv/bin/activate

# Install PyTorch CUDA 12.4
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### CPU only

```bash
# Create venv
python3.12 -m venv .venv
source .venv/bin/activate

# Install PyTorch CPU (with MKL for optimal performance)
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

---

## 🪟 Windows

### AMD GPU (ROCm 7.2)

Requires **AMD Adrenalin 26.2.2+** — see [AMD GPU Setup](gpu-setup-amd.md) first.

```bat
:: Create venv
py -3.12 -m venv .venv
.venv\Scripts\activate.bat

:: Install ROCm SDK
pip install --no-cache-dir ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_core-7.2.0.dev0-py3-none-win_amd64.whl ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_devel-7.2.0.dev0-py3-none-win_amd64.whl ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_libraries_custom-7.2.0.dev0-py3-none-win_amd64.whl ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm-7.2.0.dev0.tar.gz

:: Install PyTorch ROCm 7.2
pip install --no-cache-dir ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torch-2.9.1+rocmsdk20260116-cp312-cp312-win_amd64.whl ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torchaudio-2.9.1+rocmsdk20260116-cp312-cp312-win_amd64.whl ^
    https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torchvision-0.24.1+rocmsdk20260116-cp312-cp312-win_amd64.whl

:: Install dependencies
pip install -r requirements.txt

:: Install package
pip install -e .
```

> ⚠️ The AMD ROCm repository URLs may change with new releases. Check [repo.radeon.com/rocm/windows](https://repo.radeon.com/rocm/windows/) for the latest wheel paths.

### NVIDIA GPU (CUDA)

> ⚠️ CUDA support has not been tested by the maintainers — no NVIDIA hardware was available for testing. Community feedback welcome.

```bat
:: Create venv
py -3.12 -m venv .venv
.venv\Scripts\activate.bat

:: Install PyTorch CUDA 12.4
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 ^
    --extra-index-url https://download.pytorch.org/whl/cu124

:: Install dependencies
pip install -r requirements.txt

:: Install package
pip install -e .
```

### CPU only

```bat
:: Create venv
py -3.12 -m venv .venv
.venv\Scripts\activate.bat

:: Install PyTorch CPU (with MKL for optimal performance)
pip install torch torchvision torchaudio ^
    --index-url https://download.pytorch.org/whl/cpu

:: Install dependencies
pip install -r requirements.txt

:: Install package
pip install -e .
```

---

## Key Version Notes

| Backend | PyTorch | Notes |
|---------|---------|-------|
| AMD Linux (ROCm 6.4) | 2.9.1+rocm6.4 | Tested ✅ |
| AMD Windows (ROCm 7.2) | 2.9.1+rocmsdk20260116 | Tested ✅ |
| NVIDIA CUDA 12.4 | 2.5.1+cu124 | Not tested ⚠️ |
| CPU | latest stable | Tested ✅ |

> ⚠️ `ultralytics==8.4.9` is pinned in `requirements.txt` — do not upgrade without testing, as newer versions may break compatibility.
