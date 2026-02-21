import os
import sys
from pathlib import Path

# Radice assoluta del progetto
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Aggiungiamo 'src' al path di sistema a runtime per sicurezza
sys.path.append(str(PROJECT_ROOT / "src"))

# Costanti per le cartelle
CONFIG_DIR = PROJECT_ROOT / "config"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR   = PROJECT_ROOT / "logs"
TEMP_DIR   = PROJECT_ROOT / "temp_workdir"
# Nuova cartella per i check periodici (es. pulizia lenti)
CHECKS_DIR = PROJECT_ROOT / "checks"
# File per la cache delle coordinate astronomiche
COORDS_CACHE_JSON = CONFIG_DIR / "coords_cache.json"

# File specifici
CAMERAS_JSON = CONFIG_DIR / "cameras.json"
SETTINGS_JSON = CONFIG_DIR / "settings.json"
PROMPTS_JSON = CONFIG_DIR / "prompts.json"
CLIP_BLIP_JSON = CONFIG_DIR / "clip_blip_settings.json"

# Verifica ambiente
HAS_OLLAMA = os.system("command -v ollama > /dev/null") == 0
# Assicuriamoci che le cartelle base esistano sempre
for folder in [CONFIG_DIR, MODELS_DIR, LOGS_DIR,CHECKS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)