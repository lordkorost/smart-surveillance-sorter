# 🛡️ Smart Surveillance Sorter

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![Status](https://img.shields.io/badge/status-stable-green.svg)
![Hardware](https://img.shields.io/badge/HW-CUDA%20%7C%20ROCm%20%7C%20CPU-orange.svg)
![AI](https://img.shields.io/badge/AI-YOLO%20%7C%20CLIP%20%7C%20BLIP%20%7C%20Vision-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

Organize your NVR videos using the power of YOLO, CLIP, BLIP, and Vision models (Ollama).
Designed for those overwhelmed by hundreds of useless recordings caused by wind, insects, or leaves, this tool scans every video and automatically categorizes it into: **PERSON**, **ANIMAL**, **VEHICLE**, or **OTHERS**.

---

## 📋 Table of Contents

- [✨ Features](#-key-features)
- [🔧 Requirements](#-requirements)  
- [🚀 Quick Start](#-quick-start)
- [📦 Installation](#-installation)
- [📖 Documentation](#-documentation)
- [📊 Benchmarks](#-benchmarks)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

## 📖 Documentation

- [Scanning Logic & Early Exit](docs/scanning-logic.md)
- [Tuning Guide](docs/tuning-guide.md)
- [Windows GPU Setup](docs/windows-gpu.md)
- [Benchmarks](docs/benchmarks.md)
- [Camera Configuration](docs/cameras-config.md)

## ✨ Features

- **Hybrid Pipeline** – YOLO for speed → CLIP+BLIP for precision → Vision (Ollama) for uncertain cases (optional).
- **Highly Customizable** – Fine-tune settings to adapt to any camera, scenario, or environment.
- **Web UI** – An intuitive Gradio-based interface to configure and launch the sorter in all modes.
- **Test Mode** – Built-in sandbox to verify your configuration before running it on real production folders.
- **Resilient** – Automatic resume at every stage. If the power goes out, it picks up exactly where it left off.
- **Cumulative Archive** – Use a fixed output folder across multiple runs to automatically build a permanent categorized archive across days, weeks, or months.
- **Real-Time & Batch** – Works on historical archives or in constant monitoring mode as your NVR saves new files.
- **Intelligent Early Exit** – A smart mechanism that stops video analysis once a detection is confirmed, saving significant time.
- **Total Privacy** – Runs 100% locally. No data ever leaves your network.
- **Lens Cleanliness Check** – Automatically monitors camera lens status using Vision models (Ollama).

---

## 🚀 Quick Start

### 🛠️ Requirements

- **Python 3.12** — required (both Linux and Windows)
- **RAM:** 12GB minimum, 16GB recommended
- **VRAM:** 8GB minimum for GPU mode, 12GB+ for Vision/Ollama
- **Ollama** — required for Vision mode (recommended model: `qwen3-vl:8b`)

> ℹ️ CPU mode works on any modern system but is significantly slower — see [Benchmarks](#-benchmarks).

---

### 📦 Installation

#### 1. Clone the repository
```bash
git clone https://github.com/your-username/smart-surveillance-sorter.git
cd smart-surveillance-sorter
```

#### 2. Run the installer

**Linux:**
```bash
chmod +x install.sh
./install.sh --use-rocm    # AMD GPU
./install.sh --use-cuda    # NVIDIA GPU
./install.sh --use-cpu     # CPU only
```

**Windows:**
```bat
.\install.bat --use-rocm     :: AMD GPU (ROCm 7.2)
.\install.bat --use-cuda     :: NVIDIA GPU
.\install.bat --use-cpu      :: CPU only
```

> ℹ️ Windows requires Python 3.12 installed and added to PATH.  
> Download: https://www.python.org/downloads/release/python-31212/

#### 3. Launch
```bash
./run.sh      # Linux
.\run.bat     # Windows
```



> ℹ️ **CPU mode** works out of the box on any modern system — just run `./install.sh --use-cpu` (Linux) or `.\install.bat --use-cpu` (Windows). No additional drivers required. PyTorch is installed with MKL for optimal CPU performance. See [Benchmarks](#-benchmarks) for expected processing times.
---


### 🖥️ GPU Setup

| GPU | Linux | Windows |
|-----|-------|---------|
| NVIDIA | CUDA 12.x driver | CUDA 12.x driver |
| AMD | ROCm 7.2 | Adrenalin 26.1.2 + Ollama version from AMD Adrenalin - AI Bundle - Ollama |

> 📖 **AMD GPU detailed setup:** [docs/gpu-setup-amd.md](docs/gpu-setup-amd.md)

> ⚠️ **AMD GPU on Windows + Ollama (Vision mode):** Set `OLLAMA_VULKAN=1` as a system environment variable to enable GPU acceleration. Without this, Ollama runs on CPU only.

#### **CLI Usage**
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
> Always use the "Test Mode" first! 
> Before letting the sorter move your real NVR recordings, run it with the --test flag (or enable "Test Mode" in the Web UI). In this mode, the software will copy files instead of moving them, allowing you to verify if the detection and categorization are working as expected for your specific camera angles.


## 📊 Benchmarks

**Test cameras:** Reolink 4K
- Daytime: 20 fps
- Nighttime: 12 fps
- Resolution: 3840×2160

> ℹ️ Parameters tuned for Reolink 4K footage. Other cameras with similar specs (4K, 12-20fps) should work well with default parameters. Lower resolution or fps cameras may need stride/occurrence adjustments — see [YOLO Tuning](docs/benchmarks/yolo-tuning.md).

Tested on **521 videos + 480 images** (1 day of NVR footage, 8 cameras, mixed outdoor scenes).  
Hardware: Ryzen 5 9600X | RX 9060 XT 16GB | ROCm 6.4 (Linux) / Vulkan (Windows)

> 📖 Full benchmark details: [docs/benchmarks.md](docs/benchmarks.md)

### Performance

| Mode | Linux GPU | Windows GPU | Linux CPU | Windows CPU |
|------|-----------|-------------|-----------|-------------|
| YOLO only (img) | 00:25 | 00:30 | 00:52 | 01:02 |
| YOLO only (vid) | 42:15 | 42:43 | 01:00:14 | 01:13:06 |
| +BLIP | 02:51 | 03:09 | 07:50 | 05:10 |
| +BLIP+Fallback | 02:51 | 07:26 | — | — |
| +Vision | 15:55 | 38:12 | — | — |

> ⚠️ **Timings depend on your footage characteristics:**
> - **YOLO**: scales with video length — longer videos = more frames to analyze. Test set: ~30% short clips (25-30s), ~40% medium (1 min), ~30% long (3+ min).
> - **Vision**: varies significantly by scene complexity — ambiguous scenes (shadows, partial objects, night) trigger longer AI reasoning. Simple scenes can be as fast as ~1-2s/video, complex ones up to ~15s/video.
> - **BLIP**: largely unaffected by video length — processes only extracted keyframes.

### ⏱️ Total Pipeline Time (521 videos + 480 images)

| Pipeline | Linux GPU | Windows GPU | Linux CPU | Windows CPU |
|----------|-----------|-------------|-----------|-------------|
| YOLO+BLIP | ~46 min | ~47 min | ~1h 09 min | ~1h 19 min |
| YOLO+BLIP+Fallback | ~48 min | ~54 min | — | — |
| YOLO+Vision | ~58 min | ~1h 21 min | — | — |

> ℹ️ CPU times measured with standard PyTorch build (MKL). ROCm build on CPU is significantly slower — see [docs/benchmarks.md](docs/benchmarks.md).

### Accuracy (YOLO + BLIP, default params)

> 📖 **Precision** = of all videos classified as X, how many were actually X (false positive rate).  
> **Recall** = of all real X videos, how many were correctly found (false negative rate).  
> **Global accuracy** = percentage of correctly classified videos overall.

| Category | Precision | Recall |
|----------|-----------|--------|
| PERSON | 95.9% | 100.0% |
| VEHICLE | 100.0% | 91.7% |
| ANIMAL | 95.2% | 76.9% |
| **Global accuracy** | **98.27%** | |

### Accuracy comparison by mode

| Mode | Global Acc | Avg Recall | Notes |
|------|------------|------------|-------|
| YOLO+BLIP | 98.27% | 89.53% | Fast, recommended default |
| YOLO+BLIP+Fallback | 97.89% | 88.25% | ⚠️ May worsen results |
| YOLO+Vision | 98.46% | 91.92% | Best accuracy, slower |

> 🎯 **0 missed persons (FN=0)** across all test runs — the system never fails to detect a real person. False positives (shadows, reflections) are filtered by the Vision refinement step.

Per il README li puoi citare come known limitations:

⚠️ Partial detections (person visible only through glass or partially behind obstacles) may produce inconsistent results depending on lighting and angle. YOLO may detect the person while Vision cannot confirm from the full frame.

## 🤝 Contributing

Contributions are welcome! If you want to improve the project:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'feat: add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

### Areas where help is appreciated
- NVIDIA GPU testing and validation
- Windows compatibility improvements  
- New NVR filename template support
- Performance optimizations
- Documentation translations

Please open an Issue before starting major changes to discuss the approach.


## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.



