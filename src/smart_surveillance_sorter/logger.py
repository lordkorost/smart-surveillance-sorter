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
    """Formatter for colored console logging output with custom formatting rules."""
    # ANSI color palette
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
        """Format log record with colors and highlighting for different message components."""
        # 1. Time and Log Level
        log_time = time.strftime("%H:%M:%S", time.localtime(record.created))
        colored_time = f"{self.GREEN}{log_time}{self.RESET}"
        level_color = self.COLORS.get(record.levelname, self.RESET)
        colored_level = f"{level_color}{record.levelname:<7}{self.RESET}"

        # 2. Message
        msg = record.getMessage()

        # --- COLORING RULES ---
        
        # A. FILE (mp4, jpg, etc) - Do this first so it's not overwritten
        msg = re.sub(r'([\w\-_]+\.(mp4|jpg|png|json|txt|pt))', f"{self.YELLOW}\\1{self.RESET}", msg)

        # B. TRIGGER '=' (Key=Value)
        # This version also accepts values with spaces (like "NVR reo_02...")
        # until it encounters a comma, pipe, or arrow ->
        msg = re.sub(
            r'\b([\w\-_]+)=([^,|>\n]+)', 
            f"{self.CYAN}\\1{self.RESET}={self.YELLOW}\\2{self.RESET}", 
            msg
        )

        # C. BOOLEANS AND NUMBERS (only if not already colored)
        msg = re.sub(r'(?<![\x1b=])\b(True|False|\d+\.?\d*)\b', f"{self.YELLOW}\\1{self.RESET}", msg)

        # D. ARROWS
        msg = msg.replace("->", f"{self.CYAN}->{self.RESET}")

        # 3. WHITE only at the end for remaining uncolored parts
        return f"{colored_time} {colored_level}: {self.WHITE}{msg}{self.RESET}"

def get_logger(name: str = None, debug: bool = False) -> logging.Logger:
    logger = logging.getLogger(name)
    
    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)

    # 1. Handler for Console (Always active)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(ColorFormatter())
    logger.addHandler(ch)

    # 2. Handler for File (Only if debug is True)
    if debug:
        os.makedirs(LOGS_DIR, exist_ok=True)

        log_filename = LOGS_DIR / "debug_session.log"
        
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Rotating configuration:
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

    # Suppress system warnings (Torch/AMD/Flash Attention/GELU)
    import warnings
    warnings.filterwarnings("ignore")

    logger.propagate = False
    return logger

# ------------------------------------------------------------------
# 1.1  CPU
# ------------------------------------------------------------------
def get_cpu_usage() -> float:
    """Get CPU usage percentage using psutil.
    
    Returns:
        CPU usage as percentage value
    """
    return psutil.cpu_percent(interval=0.1)

def get_ram_info() -> Dict[str, float]:
    """Get RAM statistics: total, used, free in GiB.
    
    Returns:
        Dictionary with 'total', 'used', 'free' keys in GiB
    """
    vm = psutil.virtual_memory()
    return {
        "total": vm.total      / (1024 ** 3),
        "used":  vm.used       / (1024 ** 3),
        "free":  vm.available  / (1024 ** 3),
    }


def get_gpu_info() -> Dict[str, float]:
    """Detect VRAM universally:
    1. AMD Linux via sysfs
    2. AMD/ROCm or NVIDIA via torch
    3. NVIDIA via GPUtil
    4. Fallback to 0.0
    
    Returns:
        Dictionary with 'total', 'used', 'free' VRAM in GB
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
# Detect device (CPU vs CUDA)
# ------------------------------------------------------------------
def detect_device() -> Tuple[Union[str, None], Union[torch.device, None]]:
    """Detect available device for model inference.
    
    Returns:
        Tuple of (device_string, torch_device):
        - 'cuda' or 'cpu' (None if CUDA not available)
        - torch.device('cuda:0') or torch.device('cpu')
    """
    if torch.cuda.is_available():
        return "cuda", torch.device("cuda:0")
    else:
        return "cpu", torch.device("cpu")
    

def get_system_stats() -> Dict:
    """Retrieve comprehensive system resource usage statistics.
    
    Returns a dictionary containing:
        - cpu_usage: CPU usage percentage
        - ram_total/ram_used/ram_free: RAM statistics in GB
        - vram_total/vram_used: VRAM statistics in GB (if CUDA available)
    """
    vm = psutil.virtual_memory()
    stats = {
        "cpu_usage": psutil.cpu_percent(interval=None),
        "ram_total": vm.total / (1024**3),
        "ram_used":  vm.used  / (1024**3),
        "ram_free":  vm.available / (1024**3),
        "vram_total": 0.0,
        "vram_used":  0.0,
    }

    # VRAM check — works on Windows and Linux (AMD and NVIDIA)
    if torch.cuda.is_available():
        try:
            free, total = torch.cuda.mem_get_info(0)
            stats["vram_total"] = total / (1024**3)
            stats["vram_used"]  = (total - free) / (1024**3)
        except Exception:
            pass


    return stats



def log_resource_usage(log: logging.Logger, prefix: str = "STATS"):
    """Log current system resource usage (CPU, RAM, VRAM) to logger.
    
    Args:
        log: Logger instance to write to
        prefix: Prefix label for the log message (default: "STATS")
    """
    s = get_system_stats()
    
    # Formattazione in una singola riga
    msg = (
        f"[{prefix}] "
        f"CPU: {s['cpu_usage']:>4.1f}% | "
        f"RAM: {s['ram_used']:.1f}/{s['ram_total']:.1f}GB | "
        f"VRAM: {s['vram_used']:.1f}/{s['vram_total']:.1f}GB"
    )
    log.info(msg)


# ANSI colors to match tqdm with logger
LOG_GREEN = "\033[92m"
LOG_RESET = "\033[0m"

def get_pbar_prefix(desc: str = "Progress") -> str:
    """Generate colored prefix with timestamp for tqdm progress bar.
    
    Args:
        desc: Description text for the progress bar
        
    Returns:
        Formatted prefix string with timestamp and color codes
    """
    ts = time.strftime('%H:%M:%S')
    return f"{LOG_GREEN}{ts} INFO   {LOG_RESET}: {desc}"
