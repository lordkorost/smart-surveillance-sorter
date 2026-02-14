from datetime import datetime
import logging
import os
from pathlib import Path
import sys
import time


def configure_logger(debug: bool = False, log_file_name: str = "analysis.log"):
    """
    Configure the *root* logger.  
    Call this once per process (main.py, web‑ui.py, uvicorn, etc.).
    """
    root = logging.getLogger()
    if root.handlers:
        # Already configured
        return

    level = logging.DEBUG if debug else logging.INFO
    root.setLevel(level)
    
    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(_color_formatter(level))
    # formatter che stampa prima l’orario
    #ch.setFormatter(logging.Formatter(fmt="%(asctime)s %(levelname)s: %(message)s",datefmt="%Y-%m-%d %H:%M:%S"))
    root.addHandler(ch)

    # Silenzia i log delle librerie HTTP
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    # Se vuoi silenziare anche i messaggi di caricamento di HuggingFace
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    # Silenzia il rumore delle librerie HTTP (anche il debug profondo)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    # Silenzia HuggingFace e i suoi componenti di caching/download
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)
    # Optional file
    # 2. File Handler (SOLO se debug è True)
    if debug:
        from pathlib import Path
        from datetime import datetime
        
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_log_path = log_dir / f"{timestamp}_{log_file_name}"

        fh = logging.FileHandler(full_log_path, mode='a', encoding='utf-8')
        fh.setLevel(logging.DEBUG) # Scriviamo tutto nel file
        fh.setFormatter(_plain_formatter())
        root.addHandler(fh)
        
        
        
    # Prevent duplication if imported in sub‑modules
    root.propagate = False

# Helpers for formatters ---------------------------------------------------
def _plain_formatter():
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    return logging.Formatter(fmt, "%Y-%m-%d %H:%M:%S")

def _color_formatter(level):
    class ColouredFormatter(logging.Formatter):
        colours = {
            logging.DEBUG: '\x1b[34m',    # Blu
            logging.INFO: '\x1b[32m',     # Verde
            logging.WARNING: '\x1b[33m',  # Giallo
            logging.ERROR: '\x1b[31m',    # Rosso
            logging.CRITICAL: '\x1b[35m', # Viola
        }
        reset = '\x1b[0m'

        def format(self, record):
            log_fmt = f"{self.colours.get(record.levelno)}%(asctime)s %(levelname)s: %(message)s{self.reset}"
            formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
            return formatter.format(record)

    return ColouredFormatter()

def _json_formatter():
    import json
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            d = {
                "time": record.created,
                "level": record.levelname,
                "name": record.name,
                "msg": record.getMessage(),
            }
            return json.dumps(d)
    return JsonFormatter()