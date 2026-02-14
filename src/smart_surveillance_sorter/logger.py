# logger.py
import logging
import os
import time
import psutil
import GPUtil
from datetime import datetime
from typing import Dict, Optional

# ------------------------------------------
# 1. Formattatore (timestamp + livello + messaggio)
# ------------------------------------------
class ColorFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[94m',      # blue
        'INFO': '\033[92m',       # green
        'WARNING': '\033[93m',    # yellow
        'ERROR': '\033[91m',      # red
        'CRITICAL': '\033[95m',   # magenta
        'RESET': '\033[0m',
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, '')
        reset = self.COLORS['RESET']
        fmt = f"{color}[{record.levelname}] {record.asctime} {record.message}{reset}"
        record.message = fmt
        return super().format(record)


def get_logger(name: str = __name__, debug: bool = False) -> logging.Logger:
    """
    Returns a logger that prints to console.
    If `debug=True` the level is set to DEBUG, otherwise INFO.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)

    fmt = "%(message)s"
    formatter = ColorFormatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")
    formatter.converter = time.gmtime   # to use UTC in timestamps

    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Prevent duplicate messages in child modules
    logger.propagate = False
    return logger

def get_cpu_usage() -> float:
    """Percentuale di CPU occupata."""
    return psutil.cpu_percent(interval=0.1)

def get_ram_usage() -> Dict[str, float]:
    """RAM totale, usata, libera in GiB."""
    vm = psutil.virtual_memory()
    return {
        "total": vm.total / (1024 ** 3),
        "used": vm.used / (1024 ** 3),
        "free": vm.available / (1024 ** 3),
    }

def get_gpu_usage() -> Dict[str, float]:
    """VRAM totale, usata, libera per la GPU più occupata."""
    gpus = GPUtil.getGPUs()
    if not gpus:
        return {"total": 0, "used": 0, "free": 0}
    gpu = max(gpus, key=lambda g: g.memoryUtil)  # scegli quella più occupata
    total = gpu.memoryTotal
    used = gpu.memoryUsed
    return {"total": total, "used": used, "free": total - used}