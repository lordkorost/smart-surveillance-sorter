# 🛡️ Smart Surveillance Sorter

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![Status](https://img.shields.io/badge/status-active--main-green.svg)
![Hardware](https://img.shields.io/badge/HW-CUDA%20%7C%20ROCm%20%7C%20CPU-orange.svg)
![AI](https://img.shields.io/badge/AI-YOLO%20%7C%20CLIP%20%7C%20BLIP%20%7C%20Vision-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

Organize your NVR videos using the power of YOLO, CLIP, BLIP, and Vision models (Ollama).
Designed for those overwhelmed by hundreds of useless recordings caused by wind, insects, or leaves, this tool scans every video and automatically categorizes it into: **PERSON**, **ANIMAL**, **VEHICLE**, or **OTHERS**.

---

## ✨ Features

- **Hybrid Pipeline** – YOLO for speed → CLIP+BLIP for precision → Vision (Ollama) for uncertain cases (optional).
- **Highly Customizable** – Fine-tune settings to adapt to any camera, scenario, or environment.
- **Web UI** – An intuitive Gradio-based interface to configure and launch the sorter in all modes.
- **Test Mode** – Built-in sandbox to verify your configuration before running it on real production folders.
- **Intelligent Early Exit** – A smart mechanism that stops video analysis once a detection is confirmed, saving significant time.
- **Real-Time & Batch** – Works on historical archives or in constant monitoring mode as your NVR saves new files.
- **Total Privacy** – Runs 100% locally. No data ever leaves your network.
- **Resilient** – Automatic resume at every stage. If the power goes out, it picks up exactly where it left off.
- **Lens Cleanliness Check** – Automatically monitors camera lens status using Vision models (Ollama).

---

## 🚀 Quick Start

### 🛠️ Hardware Requirements

#### 💻 CPU-Only Mode (Low Resources)
*Ideal for home servers or PCs without a dedicated/supported GPU.*
* **Processor:** Intel Core i5/i7 (8th Gen+) or AMD Ryzen 5+.
* **RAM:** 8-16 GB.
* **Performance:** Excellent for batch scanning or real-time monitoring with a few cameras.
* **Models:** Runs the standard pipeline (YOLO + CLIP/BLIP).

#### ⚡ GPU Accelerated Mode (High Performance)
*Ideal for massive processing, multi-camera setups, or advanced Vision models.*
* **NVIDIA:** CUDA 12.x support.
* **AMD:** ROCm 6.4 support (Tested on RX 9600 XT).
* **VRAM:** 8GB (Base) / 12GB+ (Vision/Ollama).
* **Advantages:** Ultra-fast analysis and support for heavy Vision models.

### ⚙️ Software and Drivers
* **Python:** 3.12+ (installed and available in your user PATH).
* **Ollama:** Required for **Vision Mode** (recommended model: `qwen2.5-vl:8b`).
* **FFmpeg:** Mandatory for frame extraction from NVR video streams.

#### GPU Drivers:
* **NVIDIA:** Driver version 550+ with CUDA 12.x support.
* **AMD:** Updated drivers with ROCm support (or AI Bundle for Windows).

> [!IMPORTANT]
> 🛡️ **Compatibility Note:** The system automatically detects available hardware. If no compatible GPU (CUDA or ROCm) is found, it will automatically fallback to **CPU mode**, ensuring functionality on any modern system.

---

### 📦 Installation



Readme install · MD
Copia

#### 1. Clone the repository
```bash
git clone https://github.com/your-username/smart-surveillance-sorter.git
cd smart-surveillance-sorter
```

#### 2. Run the installer

The installer automatically creates a virtual environment and installs the correct version of PyTorch for your hardware.

### Linux (Ubuntu/Debian)

```bash
chmod +x install.sh
./install.sh
```

Optional: force a specific backend if auto-detection fails:

```bash
./install.sh --use-cuda    # NVIDIA GPU
./install.sh --use-rocm    # AMD GPU (ROCm 6.4)
./install.sh --use-cpu     # CPU only
```

Launch the application:

```bash
./run.sh
```

---

### Windows 11

> [!IMPORTANT]
> Windows requires **Python 3.12** installed and added to PATH.
> Download from: https://www.python.org/downloads/release/python-31212/

```bat
install.bat
```

Optional: force a specific backend:

```bat
install.bat --use-cuda     :: NVIDIA GPU
install.bat --use-rocm     :: AMD GPU (ROCm 7.2 for Windows)
install.bat --use-cpu      :: CPU only
```

Launch the application:

```bat
run.bat
```

> [!NOTE]
> **AMD GPU on Windows**: Requires AMD Adrenalin driver **26.1.1 or later**.
> Download from AMD Adrenalin Software → Optional → AI Bundle → PyTorch.
> Tested on RX 9060 XT with ROCm 7.2 — full GPU acceleration works.
> [!IMPORTANT]
> **AMD GPU on Windows — Ollama**: ROCm is not yet fully supported for Ollama on Windows.
> To enable GPU acceleration you must set the `OLLAMA_VULKAN=1` environment variable.
>
> **Permanent setup:**
> 1. Open "Environment Variables" in Windows Settings
> 2. Under "System variables" → New
> 4. Restart Ollama from the tray icon
>
> Without this setting Ollama will use CPU only (100% CPU, very slow).
> Tested on RX 9060 XT with AMD Adrenalin 26.1.1 + ROCm 7.2.
> [!NOTE]
> **NVIDIA GPU on Windows**: Tested with CUDA 12.4. Should work out of the box with `--use-cuda`.

> [!TIP]
> If auto-detection fails, always use the explicit `--use-rocm` or `--use-cuda` flag.
> See `docs/windows_gpu.md` for detailed GPU setup instructions.

#### **Advanced CLI Usage**
If you prefer using the terminal, check out our [**CLI Reference Guide**](docs/cli_reference.md) for all available flags and examples.

### ⚙️ Configuration

Before running the sorter, you need to set up your environment. You can do this via the Web UI or by manually editing the files in the config/ folder.
* Set your Location: Open config/settings.json and set your city. This is required to calculate sunrise/sunset times for accurate day/night detection.
* Filename Template: Ensure the filename_template in settings.json matches how your NVR saves files (e.g., CameraName_YYYYMMDD_HHMMSS.mp4). This allows the sorter to find files correctly.
```
Reolink (default)
"filename_template": "{nvr_name}_{camera_id}_{timestamp}"

Hikvision (es: CH01_20260228063426.mp4)
"filename_template": "CH{camera_id}_{timestamp}"

Dahua (es: 2026-02-28_06-34-26_cam1.mp4)
"timestamp_format": "%Y-%m-%d_%H-%M-%S",
"filename_template": "{timestamp}_{nvr_name}{camera_id}"
```
* Cameras Setup: Define your cameras in config/cameras.json. You can use [**cameras_example.json**](docs/cameras_setting.md) as a template.

#### 🛠️ Deep Dive:
* For a detailed explanation of all settings, see [**Advanced Configuration Guide**](docs/advanced_conf_guide.md)
* To learn how to handle specific edge cases (like garden gnomes or moving leaves), check [**Tuning & False Positives Guide**](docs/setting_tips.md)

⚠️ IMPORTANT: Test Before Use!
>[!CAUTION]
> Always use the "Test Mode" first! > Before letting the sorter move your real NVR recordings, run it with the --test flag (or enable "Test Mode" in the Web UI). In this mode, the software will copy files instead of moving them, allowing you to verify if the detection and categorization are working as expected for your specific camera angles.


##  📊 Test Results (Folder1: 565 Videos)

| Category | True Positives | False Positives | False Negatives | 
| :--- | :---: | :---: | :---: |
| **PERSON** | 169 | 6 | 0 |
| **VEHICLE** | 16 | 0 | 0 |
| **ANIMAL** | 21 | 0 | 9 | 
| **OTHERS** | 286 | 0 | 0 |

### ⏱️ Performance Benchmarks TODO
| Hardware | Pipeline | Avg Speed (clip lenght 25secs - 3min) |
| :--- | :--- | :--- |
| **AMD RX 9060 XT 16 GB** | YOLO + CLIPBLIP | ~1.2s |
| **CPU Ryzen 5 9600x** | YOLO + CLIP | ~10.5s |

### 📈 Detailed Analysis: 
* For more hardware benchmarks and test results check out our [**Full Test Report**](docs/tests.md)




