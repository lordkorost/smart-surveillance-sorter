import os
import sys
from pathlib import Path

# Radice assoluta del progetto
PROJECT_ROOT = Path(__file__).resolve().parents[2]


sys.path.append(str(PROJECT_ROOT / "src"))

# Costanti per le cartelle
CONFIG_DIR = PROJECT_ROOT / "config"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR   = PROJECT_ROOT / "logs"
TEMP_DIR   = PROJECT_ROOT / "temp_workdir"

CHECKS_DIR = PROJECT_ROOT / "checks"

COORDS_CACHE_JSON = CONFIG_DIR / "coords_cache.json"
#resfiles
FRAME_DIR = "extracted_frames"
YOLO_CACHE ="yolo_scan_res.json"
LENS_HEALTH = "lens_health.json"
VISION_CACHE = "vision_scan_res.json"
CLIPBLIP_CACHE = "clip_blip_res.json"
CLIPBLIP_FALLBACK_CACHE = "clip_blip_fallback_res.json"
FINAL_REPORT = "classification_results.json"
GROUND_TRUTH ="ground_truth.json"

CAMERAS_JSON = CONFIG_DIR / "cameras.json"
SETTINGS_JSON = CONFIG_DIR / "settings.json"
PROMPTS_JSON = CONFIG_DIR / "prompts.json"
CLIP_BLIP_JSON = CONFIG_DIR / "clip_blip_settings.json"


HAS_OLLAMA = os.system("command -v ollama > /dev/null") == 0

for folder in [CONFIG_DIR, MODELS_DIR, LOGS_DIR,CHECKS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)