# AMD GPU Setup

This guide covers AMD GPU driver installation for Smart Surveillance Sorter on Linux and Windows.

>[!NOTE]
> **AMD ROCm is officially supported only for specific AMD GPUs.** Check the [official AMD ROCm compatibility list](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) before proceeding.

---

## Linux

### Step 1 — Install ROCm drivers

#### Ubuntu 24.04

```bash
sudo apt update
wget https://repo.radeon.com/amdgpu-install/7.2/ubuntu/noble/amdgpu-install_7.2.70200-1_all.deb
sudo apt install ./amdgpu-install_7.2.70200-1_all.deb
sudo amdgpu-install -y --usecase=graphics,rocm
sudo usermod -a -G render,video $LOGNAME
```

#### Ubuntu 22.04

```bash
sudo apt update
wget https://repo.radeon.com/amdgpu-install/7.2/ubuntu/jammy/amdgpu-install_7.2.70200-1_all.deb
sudo apt install ./amdgpu-install_7.2.70200-1_all.deb
sudo amdgpu-install -y --usecase=graphics,rocm
sudo usermod -a -G render,video $LOGNAME
```
>[!NOTE]
> The only difference between Ubuntu 22.04 and 24.04 is the `wget` URL (`jammy` vs `noble`).

**Reboot after installation:**
```bash
sudo reboot
```

### Step 2 — Verify ROCm

```bash
rocminfo | grep -A2 "Agent Type"
# Should show your GPU as a "GPU" agent

rocm-smi
# Should show your GPU with temperature and VRAM usage
```

### Step 3 — Install Smart Surveillance Sorter

```bash
git clone https://github.com/your-username/smart-surveillance-sorter.git
cd smart-surveillance-sorter
chmod +x install.sh
./install.sh --use-rocm
```

### Step 4 — Verify GPU is detected

```bash
./run.sh
```

In the terminal you should see:

```
Backend : ROCm/HIP (AMD)
GPU     : AMD Radeon RX ...
VRAM    : ✅ OK
```
>[!NOTE]
> On Linux, AMD GPUs are exposed via the CUDA compatibility layer in PyTorch — `torch.cuda.is_available()` returns `True` for ROCm GPUs. This is expected behavior.

---

### Troubleshooting — Linux

**GPU not detected after install:**
```bash
lsmod | grep amdgpu
# If empty, the driver is not loaded — check dmesg for errors
dmesg | grep amdgpu
```

**Permission denied on GPU device:**
```bash
groups $USER
# Must include "render" and "video"
# If not, re-run: sudo usermod -a -G render,video $LOGNAME
# Then log out and back in (or reboot)
```

**rocminfo not found:**
```bash
sudo apt install rocminfo
```

---

## Windows
>[!NOTE]
> **PyTorch ROCm support on Windows is not officially maintained by the PyTorch team.** It is provided by AMD directly. See [AMD's announcement](https://rocm.docs.amd.com/en/latest/) for current status.

### Step 1 — Install AMD Adrenalin driver

Download and install **AMD Adrenalin 26.2.2 or later** from the [AMD drivers page](https://www.amd.com/en/support).

>[!NOTE]
> Older Adrenalin versions do not include the ROCm runtime required for PyTorch. Version 26.2.2+ is required.

### Step 2 — Install Ollama (Vision mode only)
>[!NOTE]
> Skip this step if you don't plan to use Vision mode (`--vision` or `--fallback`).

Ollama on Windows must be installed from the **AMD Adrenalin AI Bundle** — this is the only version tested and confirmed to work with AMD GPU acceleration on Windows.

1. Open **AMD Software: Adrenalin Edition**
2. Go to **AMD Install Manager**
3. Find **AI Bundle** and install it
4. Select **Ollama** from the bundle components

>[!NOTE]
> The official Ollama installer from ollama.com is not tested and may not support AMD GPU acceleration on Windows. Use the AMD bundle version.

After installation, enable GPU acceleration:

1. Open **System Properties** → **Advanced** → **Environment Variables**
2. Under **System variables**, click **New**:
   - Variable name: `OLLAMA_VULKAN`
   - Variable value: `1`
3. Click OK and **restart your PC**

>[!NOTE]
> Ollama uses Vulkan (not ROCm) for GPU acceleration on Windows. Without `OLLAMA_VULKAN=1`, Ollama runs on CPU only — Vision mode will be very slow.

### Step 3 — Install Smart Surveillance Sorter

```bat
.\install.bat --use-rocm
```

The installer downloads and installs the ROCm SDK and PyTorch ROCm 7.2 wheels directly from AMD's repository. This is a **large download (~3-5 GB)** — be patient.

>[!NOTE]
> Run from **Command Prompt** or **PowerShell** in the project folder.

### Step 4 — Verify

```bat
.\run.bat
```

In the terminal you should see:

```
Backend : ROCm/HIP (AMD)
GPU     : AMD Radeon RX ...
VRAM    : OK
```

---

### Troubleshooting — Windows

**GPU not detected / VRAM error:**
- Confirm Adrenalin 26.2.2+ is installed — check in AMD Software → System tab
- Re-run `.\install.bat --use-rocm` to reinstall the ROCm wheels
- Make sure no other process is using the full VRAM

**Ollama still running on CPU after setting OLLAMA_VULKAN=1:**
- Confirm the variable is set as a **System** variable, not User
- Full reboot required — not just a terminal restart
- Check Ollama logs: `%APPDATA%\ollama\logs\`
- Confirm Ollama was installed from the AMD AI Bundle, not from ollama.com

**Install fails on ROCm SDK download:**
- The AMD repository URLs may change with new releases — check [repo.radeon.com](https://repo.radeon.com/rocm/windows/) for the latest wheel paths
- Try running `.\install.bat --use-rocm` again — pip will resume from cache
