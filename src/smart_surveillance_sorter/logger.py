# logger.py
import logging
from logging.handlers import RotatingFileHandler
import os
import time
import psutil
from typing import Dict, Tuple, Union
import torch

try:
    import GPUtil
except ImportError:
    GPUtil = None

import re

import torch

from smart_surveillance_sorter.constants import LOGS_DIR

class ColorFormatter(logging.Formatter):
    # Palette colori ANSI
    RESET  = "\033[0m"
    GREEN  = "\033[92m"
    PURPLE = "\033[95m"
    YELLOW = "\033[93m"
    WHITE  = "\033[37m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"   
    BLUE   = "\033[94m"   

    COLORS = {
        'DEBUG': PURPLE,
        'INFO': GREEN,
        'WARNING': YELLOW,
        'ERROR': RED,
        'CRITICAL': RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        # 1. Orario e Livello
        log_time = time.strftime("%H:%M:%S", time.localtime(record.created))
        colored_time = f"{self.GREEN}{log_time}{self.RESET}"
        level_color = self.COLORS.get(record.levelname, self.RESET)
        colored_level = f"{level_color}{record.levelname:<7}{self.RESET}"

        # 2. Messaggio 
        msg = record.getMessage()

        # --- REGOLE DI COLORAZIONE ---
        
        # A. FILE (mp4, jpg, ecc) - Lo facciamo per primo così non viene sovrascritto
        msg = re.sub(r'([\w\-_]+\.(mp4|jpg|png|json|txt|pt))', f"{self.YELLOW}\\1{self.RESET}", msg)

        # B. TRIGGER '=' (Key=Value)
        # Questa versione accetta anche valori con spazi (come "NVR reo_02...") 
        # finché non incontra una virgola, una pipe o la freccia ->
        msg = re.sub(
            r'\b([\w\-_]+)=([^,|>\n]+)', 
            f"{self.CYAN}\\1{self.RESET}={self.YELLOW}\\2{self.RESET}", 
            msg
        )

        # C. BOOLEANI E NUMERI (solo se non già colorati)
        msg = re.sub(r'(?<![\x1b=])\b(True|False|\d+\.?\d*)\b', f"{self.YELLOW}\\1{self.RESET}", msg)

        # D. FRECCE
        msg = msg.replace("->", f"{self.CYAN}->{self.RESET}")

        # 3. BIANCO solo alla fine alle parti rimaste senza colore
        return f"{colored_time} {colored_level}: {self.WHITE}{msg}{self.RESET}"

def get_logger(name: str = None, debug: bool = False) -> logging.Logger:
    logger = logging.getLogger(name)
    
    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)

    # 1. Handler per la Console (Sempre attivo)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(ColorFormatter())
    logger.addHandler(ch)

    # 2. Handler per il File (Solo se debug è True)
    if debug:
        os.makedirs(LOGS_DIR, exist_ok=True)

        log_filename = LOGS_DIR / "debug_session.log"
        
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Configurazione Rotante: 
        # maxBytes=10MB,
        fh = RotatingFileHandler(
            log_filename, 
            maxBytes=10*1024*1024, 
            backupCount=5, 
            encoding='utf-8'
        )
        
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)
        
        logger.debug(f"--- LOG DEBUG ROTATE {log_filename} (Max 5 file) ---")

    external_libs = [
        "httpx", "httpcore", "urllib3", "huggingface_hub", 
        "timm", "transformers", "open_clip", "ultralytics", 
        "onnxruntime","matplotlib"
    ]
    
    for lib_name in external_libs:
        ext_logger = logging.getLogger(lib_name)
        ext_logger.setLevel(logging.ERROR) 
        ext_logger.propagate = False       

    # Blocca i Warning di sistema (quelli di Torch/AMD/Flash Attention/GELU)
    import warnings
    warnings.filterwarnings("ignore")

    logger.propagate = False
    return logger

# ------------------------------------------------------------------
# 1.1  CPU
# ------------------------------------------------------------------
def get_cpu_usage() -> float:
    """Percentuale di CPU occupata (usando psutil)."""
    return psutil.cpu_percent(interval=0.1)

def get_ram_info() -> Dict[str, float]:
    """RAM totale, usata, libera (in GiB)."""
    vm = psutil.virtual_memory()
    return {
        "total": vm.total      / (1024 ** 3),
        "used":  vm.used       / (1024 ** 3),
        "free":  vm.available  / (1024 ** 3),
    }


def get_gpu_info() -> Dict[str, float]:
    """
    Rilevamento VRAM Universale:
    1. AMD Linux tramite sysfs
    2. AMD/ROCm o NVIDIA tramite torch
    3. NVIDIA tramite GPUtil
    4. Fallback 0.0
    """
    # --- 1. AMD Linux sysfs ---
    if os.name == "posix":  # Linux/Mac
        amd_paths = [
            "/sys/class/drm/card1/device", 
            "/sys/class/drm/renderD128/device",
            "/sys/class/drm/card0/device"
        ]
        for base_path in amd_paths:
            vram_used_p = os.path.join(base_path, "mem_info_vram_used")
            vram_total_p = os.path.join(base_path, "mem_info_vram_total")
            if os.path.exists(vram_used_p) and os.path.exists(vram_total_p):
                try:
                    with open(vram_used_p, 'r') as f:
                        used = int(f.read().strip()) / (1024**3)
                    with open(vram_total_p, 'r') as f:
                        total = int(f.read().strip()) / (1024**3)
                    return {"total": total, "used": used, "free": total - used}
                except Exception:
                    continue

    # --- 2. Torch CUDA/ROCm ---
    if torch.cuda.is_available():
        t = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        u = torch.cuda.memory_allocated(0) / (1024**3)
        return {"total": t, "used": u, "free": t - u}

    # --- 3. GPUtil NVIDIA fallback ---
    if GPUtil:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = max(gpus, key=lambda g: g.memoryUtil)
                t = gpu.memoryTotal / 1024
                u = gpu.memoryUsed / 1024
                return {"total": t, "used": u, "free": t - u}
        except Exception:
            pass

    # --- 4. Fallback CPU ---
    return {"total": 0.0, "used": 0.0, "free": 0.0}

# ------------------------------------------------------------------
# Rileva device (CPU vs CUDA)
# ------------------------------------------------------------------
def detect_device() -> Tuple[Union[str, None], Union[torch.device, None]]:
    """
    Ritorna:
      - 'cuda' o 'cpu' (o None se non è CUDA)
      - torch.device('cuda:0') oppure torch.device('cpu')
    """
    if torch.cuda.is_available():
        return "cuda", torch.device("cuda:0")
    else:
        return "cpu", torch.device("cpu")
    
# def log_device_status(log: logging.Logger, device_str: str, torch_dev: torch.device) -> None:
#     """
#     Stampa un messaggio di log (INFO) con:
#       - tipo device (cuda/cpu)
#       - VRAM (per cuda) o RAM (per cpu)
#       - Percentuale CPU (solo se cpu)
#     """
#     if device_str == "cuda":
#         gpu = get_gpu_info()
#         msg = (
#             f"🛠️ [Scanner] Initialized on device={torch_dev} | "
#             f"VRAM Total={gpu['total']:.2f} GB | "
#             f"Used={gpu['used']:.2f} GB | "
#             f"Free={gpu['free']:.2f} GB"
#         )
#     else:   # CPU
#         ram = get_ram_info()
#         cpu = get_cpu_usage()
#         msg = (
#             f"🛠️ [Scanner] Initialized on device={torch_dev} | "
#             f"RAM Total={ram['total']:.2f} GB | "
#             f"Used={ram['used']:.2f} GB | "
#             f"Free={ram['free']:.2f} GB | "
#             f"CPU={cpu:.1f}%"
#         )
#     log.info(msg)


def get_system_stats() -> Dict:
    vm = psutil.virtual_memory()
    stats = {
        "cpu_usage": psutil.cpu_percent(interval=None),
        "ram_total": vm.total / (1024**3),
        "ram_used":  vm.used  / (1024**3),
        "ram_free":  vm.available / (1024**3),
        "vram_total": 0.0,
        "vram_used":  0.0,
    }

    # VRAM — funziona su Windows e Linux (AMD e NVIDIA)
    if torch.cuda.is_available():
        try:
            free, total = torch.cuda.mem_get_info(0)
            stats["vram_total"] = total / (1024**3)
            stats["vram_used"]  = (total - free) / (1024**3)
        except Exception:
            pass


    return stats


# def get_system_stats() -> Dict:
#     """Raccoglie tutte le statistiche hardware in un unico dizionario."""
#     # 1. CPU & RAM (Universale)
#     vm = psutil.virtual_memory()
#     stats = {
#         "cpu_usage": psutil.cpu_percent(interval=None), 
#         "ram_total": vm.total / (1024 ** 3),
#         "ram_used": vm.used / (1024 ** 3),
#         "ram_free": vm.available / (1024 ** 3),
#         "gpu_load": 0.0,
#         "vram_total": 0.0,
#         "vram_used": 0.0,
#     }

#     # # 2. GPU AMD (Linux)
#     # amd_base = "/sys/class/drm/card1/device" 
#     # if not os.path.exists(amd_base):
#     #     amd_base = "/sys/class/drm/card0/device"

#     # if os.path.exists(os.path.join(amd_base, "mem_info_vram_used")):
#     #     try:
#     #         with open(os.path.join(amd_base, "mem_info_vram_used"), 'r') as f:
#     #             stats["vram_used"] = int(f.read().strip()) / (1024**3)
#     #         with open(os.path.join(amd_base, "mem_info_vram_total"), 'r') as f:
#     #             stats["vram_total"] = int(f.read().strip()) / (1024**3)
#     #         # % Utilizzo GPU AMD
#     #         busy_path = os.path.join(amd_base, "gpu_busy_percent")
#     #         if os.path.exists(busy_path):
#     #             with open(busy_path, 'r') as f:
#     #                 stats["gpu_load"] = float(f.read().strip())
#     #         return stats
#     #     except: pass

#     # # 3. GPU NVIDIA (Win/Linux)
#     # try:
#     #     gpus = GPUtil.getGPUs()
#     #     if gpus:
#     #         gpu = max(gpus, key=lambda g: g.memoryUtil)
#     #         stats["vram_total"] = gpu.memoryTotal / 1024
#     #         stats["vram_used"] = gpu.memoryUsed / 1024
#     #         stats["gpu_load"] = gpu.load * 100
#     #         return stats
#     # except: pass

#     # GPU AMD/NVIDIA Windows+Linux via torch
  

#     return stats



def log_resource_usage(log: logging.Logger, prefix: str = "STATS"):
    s = get_system_stats()
    
    # Formattazione in una singola riga
    msg = (
        f"[{prefix}] "
        f"CPU: {s['cpu_usage']:>4.1f}% | "
        f"RAM: {s['ram_used']:.1f}/{s['ram_total']:.1f}GB | "
        f"VRAM: {s['vram_used']:.1f}/{s['vram_total']:.1f}GB"
    )
    log.info(msg)


# Colori ANSI per uniformare tqdm al logger
LOG_GREEN = "\033[92m"
LOG_RESET = "\033[0m"

def get_pbar_prefix(desc: str = "Progress") -> str:
    """Genera il prefisso colorato con timestamp per tqdm."""
    ts = time.strftime('%H:%M:%S')
    return f"{LOG_GREEN}{ts} INFO   {LOG_RESET}: {desc}"
