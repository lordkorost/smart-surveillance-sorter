# 🛡️ Smart Surveillance Sorter

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![Status](https://img.shields.io/badge/status-active--main-green.svg)
![Hardware](https://img.shields.io/badge/HW-CUDA%20%7C%20ROCm-orange.svg)
![AI](https://img.shields.io/badge/AI-YOLO%20%7C%20CLIP%20%7C%20BLIP%20%7C%20Vision-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)


Organizza i video del tuo NVR usando la potenza di YOLO, CLIP, BLIP e modelli Vision (Ollama).
Pensato per chi riceve centinaia di registrazioni inutili dovute a vento, insetti o foglie, questo tool esamina ogni video e lo cataloga automaticamente in: PERSON, ANIMAL, VEHICLE o OTHERS.

---

## Features

- **Pipeline Ibrida** - YOLO per la velocità → CLIP+BLIP per la precisione → Vision (Ollama) per i casi dubbi (opzionale).
- **Altamente personalizzabile** - Configurabile per adattarsi a qualsiasi telecamera e scenario.
- **Webui** - Intuitiva webui che permette di configurare e avviare il sorter in tutte le sue modalità.
- **Modalità test** - Integrato tutto ciò che serve per effettuare i test prima di usarlo su cartelle reali.
- **Early Exit Intelligent** -  Un meccanismo che permette di interrompere prima l'analisi di un video risparmiando tempo.
- **Real-Time & Batch** - Funziona sia su cartelle storiche che in monitoraggio costante mentre l'NVR salva i file.
- **Privacy Totale** - Tutto gira in locale. Nessun dato viene inviato all'esterno.
- **Resiliente** - Resume automatico in ogni fase. Se la corrente salta, riparte esattamente da dove si era fermato.
- **Check Pulizia Lenti** - Controllo dellla pulizia delle telecamere usando Vision (Ollama).

---

## 🚀 Quick Start

### 🛠️ Hardware Requirements

#### 💻 Modalità CPU-Only (Low Resources)
*Ideale per server domestici, PC senza scheda video dedicata o non supportata.*
* **Processore:** Intel Core i5/i7 (8ª gen+) o AMD Ryzen 5+.
* **RAM:** 8-16 GB.
* **Performance:** Ottima per scansione batch o real-time con poche telecamere.
* **Modelli:** Utilizza la pipeline standard (YOLO + CLIP/BLIP).

#### ⚡ Modalità GPU Accelerated (High Performance)
*Ideale per processamento massivo, molte telecamere o modelli Vision avanzati.*
* **NVIDIA:** Supporto CUDA 12.x.
* **AMD:** Supporto ROCm 6.4 (Testato su RX 9600 XT).
* **VRAM:** 8GB (Base) / 12GB+ (Vision/Ollama).
* **Vantaggi:** Analisi rapida e supporto modelli Vision pesanti.

### Software and drivers
* **Python:** installed and available in your user PATH (python3.12 tested)
* **Ollama:** Necessario se vuoi usare la modalità Vision (modello consigliato: qwen3-vl:8b).

#### GPU Drivers:
* **NVIDIA:** Driver versione 550+ con supporto CUDA 12.x.x
* **AMD:** Driver aggiornati con Rocm
* **FFmpeg:** Indispensabile per l'estrazione dei frame dai video dell'NVR.

> [!IMPORTANT]
> 🛡️ **Nota sulla compatibilità:** Il sistema rileva automaticamente l'hardware disponibile. Se non trova una GPU compatibile (CUDA o ROCm), passerà automaticamente alla modalità **CPU**, garantendo il funzionamento su qualsiasi sistema moderno.

### Installations

