# logger.py
import logging
from logging.handlers import RotatingFileHandler
import os
import time
import psutil
import GPUtil
from datetime import datetime
from typing import Dict, Optional, Tuple, Union

# # ------------------------------------------
# # 1. Formattatore (timestamp + livello + messaggio)
# # ------------------------------------------
# class ColorFormatter(logging.Formatter):
#     COLORS = {
#         'DEBUG': '\033[94m',      # blue
#         'INFO': '\033[92m',       # green
#         'WARNING': '\033[93m',    # yellow
#         'ERROR': '\033[91m',      # red
#         'CRITICAL': '\033[95m',   # magenta
#         'RESET': '\033[0m',
#     }

#     def format(self, record: logging.LogRecord) -> str:
#         color = self.COLORS.get(record.levelname, '')
#         reset = self.COLORS['RESET']
#         fmt = f"{color}[{record.levelname}] {record.asctime} {record.message}{reset}"
#         record.message = fmt
#         return super().format(record)


# def get_logger(name: str = __name__, debug: bool = False) -> logging.Logger:
#     """
#     Returns a logger that prints to console.
#     If `debug=True` the level is set to DEBUG, otherwise INFO.
#     """
#     logger = logging.getLogger(name)
#     if logger.handlers:
#         return logger  # already configured

#     level = logging.DEBUG if debug else logging.INFO
#     logger.setLevel(level)

#     # Console handler
#     ch = logging.StreamHandler()
#     ch.setLevel(level)

#     fmt = "%(message)s"
#     formatter = ColorFormatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")
#     formatter.converter = time.gmtime   # to use UTC in timestamps

#     ch.setFormatter(formatter)
#     logger.addHandler(ch)

#     # Prevent duplicate messages in child modules
#     logger.propagate = False
#     return logger

import logging
import time
import re

import torch

class ColorFormatter(logging.Formatter):
    # Palette colori ANSI
    RESET  = "\033[0m"
    GREEN  = "\033[92m"
    PURPLE = "\033[95m"
    YELLOW = "\033[93m"
    WHITE  = "\033[37m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"   # <-- AGGIUNGI QUESTA RIGA
    BLUE   = "\033[94m"   # Opzionale: utile per le chiavi se vuoi cambiare dal Cyan

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

        # 2. Messaggio - Partiamo dal testo grezzo
        msg = record.getMessage()

        # --- REGOLE DI COLORAZIONE (In ordine di priorità) ---
        
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

        # 3. Applichiamo il BIANCO solo alla fine alle parti rimaste senza colore
        # e assembliamo la riga
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
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Usiamo un nome fisso per permettere la rotazione (es: debug.log, debug.log.1, ecc.)
        log_filename = os.path.join(log_dir, "debug_session.log")
        
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Configurazione Rotante: 
        # maxBytes=10MB, tiene gli ultimi 5 file.
        fh = RotatingFileHandler(
            log_filename, 
            maxBytes=10*1024*1024, 
            backupCount=5, 
            encoding='utf-8'
        )
        
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)
        
        logger.debug(f"--- LOG DEBUG ROTANTE ATTIVATO: {log_filename} (Max 5 file) ---")

    # Blocca il rumore di fondo delle librerie esterne
    for lib in ["httpx", "httpcore", "urllib3", "huggingface_hub", "timm"]:
        logging.getLogger(lib).setLevel(logging.WARNING)

    logger.propagate = True
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

# ------------------------------------------------------------------
# 1.2  GPU (CUDA)
# ------------------------------------------------------------------
def get_gpu_info() -> Dict[str, float]:
    """VRAM totale, usata, libera (in GiB) della GPU più occupata."""
    gpus = GPUtil.getGPUs()
    if not gpus:
        return {"total": 0.0, "used": 0.0, "free": 0.0}

    # Se vuoi più di una GPU, puoi iterare e sommare
    gpu = max(gpus, key=lambda g: g.memoryUtil)  # GPU con più VRAM usata
    total = gpu.memoryTotal / 1024      # da MB a GB
    used  = gpu.memoryUsed  / 1024
    free  = total - used
    return {"total": total, "used": used, "free": free}

# ------------------------------------------------------------------
# 1.3  Rileva device (CPU vs CUDA)
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
    
def log_device_status(log: logging.Logger, device_str: str, torch_dev: torch.device) -> None:
    """
    Stampa un messaggio di log (INFO) con:
      - tipo device (cuda/cpu)
      - VRAM (per cuda) o RAM (per cpu)
      - Percentuale CPU (solo se cpu)
    """
    if device_str == "cuda":
        gpu = get_gpu_info()
        msg = (
            f"🛠️ [Scanner] Initialized on device={torch_dev} | "
            f"VRAM Total={gpu['total']:.2f} GB | "
            f"Used={gpu['used']:.2f} GB | "
            f"Free={gpu['free']:.2f} GB"
        )
    else:   # CPU
        ram = get_ram_info()
        cpu = get_cpu_usage()
        msg = (
            f"🛠️ [Scanner] Initialized on device={torch_dev} | "
            f"RAM Total={ram['total']:.2f} GB | "
            f"Used={ram['used']:.2f} GB | "
            f"Free={ram['free']:.2f} GB | "
            f"CPU={cpu:.1f}%"
        )
    log.info(msg)

# def get_cpu_usage() -> float:
#     """Percentuale di CPU occupata."""
#     return psutil.cpu_percent(interval=0.1)

# def get_ram_usage() -> Dict[str, float]:
#     """RAM totale, usata, libera in GiB."""
#     vm = psutil.virtual_memory()
#     return {
#         "total": vm.total / (1024 ** 3),
#         "used": vm.used / (1024 ** 3),
#         "free": vm.available / (1024 ** 3),
#     }

# def get_gpu_usage() -> Dict[str, float]:
#     """VRAM totale, usata, libera per la GPU più occupata."""
#     gpus = GPUtil.getGPUs()
#     if not gpus:
#         return {"total": 0, "used": 0, "free": 0}
#     gpu = max(gpus, key=lambda g: g.memoryUtil)  # scegli quella più occupata
#     total = gpu.memoryTotal
#     used = gpu.memoryUsed
#     return {"total": total, "used": used, "free": total - used}