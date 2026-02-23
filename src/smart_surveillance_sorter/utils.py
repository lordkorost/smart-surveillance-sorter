from datetime import datetime
import json
import logging
#import os
import re
#import socket
import requests
#import sys
import cv2
from pathlib import Path
from smart_surveillance_sorter.constants import PROJECT_ROOT
#from colorama import Fore, Style
import json
from astral.geocoder import database, lookup
from constants import COORDS_CACHE_JSON
from astral import Observer
from astral.sun import sun
from datetime import timedelta
import pytz
from smart_surveillance_sorter.logger import log_resource_usage
log = logging.getLogger(__name__) 
LABEL_TO_CAT = {
        "person": "PERSON", "people": "PERSON",
        "dog": "ANIMAL", "cat": "ANIMAL", "animal": "ANIMAL",
        "car": "VEHICLE", "truck": "VEHICLE", "vehicle": "VEHICLE"
    }

def load_json(full_path):
    
    if not isinstance(full_path, Path):
        full_path = PROJECT_ROOT / full_path
    #log.debug(f"📂 File da caricare {full_path}")    
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            #log.debug(f"📂 File caricato con successo: {full_path}")
            return json.load(f)
    except FileNotFoundError:
        # Usiamo il logger se disponibile, altrimenti log
        log.critical(f"Not found file={full_path}")
        return None
    except json.JSONDecodeError:
        log.critical(f"File={full_path} is not a valid JSON.")
        return None

def save_json(data, full_path):
    """
    Salva dati in un file JSON assicurandosi che la cartella esista.
    """
    if not isinstance(full_path, Path):
        full_path = PROJECT_ROOT / full_path
        
    try:
        #log.debug(f"Save file in path={full_path}")
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        log.critical(f"Error during save on path={full_path}. err={e}")
        return False

def get_video_capture(video_path):
    """Apre un video e restituisce l'oggetto cv2.VideoCapture."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning(f"Error is not possible open vid={video_path}")
        return None
    return cap


def parse_filename(path, template, ts_format):
    """
    Estrae camera_id (stringa) e timestamp (datetime) da file Reolink.
    Formato: "NVR Name_ID_YYYYMMDDHHMMSS.mp4"
    """
    stem = path.stem
    
    # Trasformiamo il template in Regex (scappiamo i caratteri speciali come _)
    # Usiamo re.escape per gestire eventuali punti o trattini nel template dell'utente
    pattern = re.escape(template)
    pattern = pattern.replace(r"\{nvr_name\}", "(?P<nvr_name>.*?)")
    pattern = pattern.replace(r"\{camera_id\}", "(?P<camera_id>.*?)")
    pattern = pattern.replace(r"\{timestamp\}", "(?P<timestamp>\\d+)")
    
    match = re.search(f"^{pattern}$", stem)
    if not match:
        return None, None
        
    try:
        data = match.groupdict()
        cam_id = data.get("camera_id")
        timestamp_str = data.get("timestamp")
        
        timestamp = datetime.strptime(timestamp_str, ts_format)
        return cam_id, timestamp
    except (ValueError, IndexError, TypeError):
        return None, None
    

def parse_filename_dynamic(file_path, storage_settings):
    """
    Estrae i metadati dal nome del file usando il template del config.
    """
    stem = file_path.stem
    # template = storage_settings["filename_template"]
    # ts_format = storage_settings["timestamp_format"]
    template = storage_settings.get("filename_template", "{camera}_{timestamp}")
    ts_format = storage_settings.get("timestamp_format", "%Y%m%d_%H%M%S")
    # Trasformiamo il template umano in Regex
    # Sostituiamo i tag con gruppi di cattura nominati
    pattern = template.replace("{nvr_name}", "(?P<nvr_name>.*?)")
    pattern = pattern.replace("{camera_id}", "(?P<camera_id>.*?)")
    pattern = pattern.replace("{timestamp}", "(?P<timestamp>\\d+)")
    
    # Aggiungiamo i limiti di inizio e fine stringa per sicurezza
    pattern = f"^{pattern}$"

    match = re.search(pattern, stem)
    if not match:
        return None, None

    try:
        data = match.groupdict()
        cam_id = data.get("camera_id")
        ts_str = data.get("timestamp")
        
        # Conversione timestamp
        timestamp = datetime.strptime(ts_str, ts_format)
        return cam_id, timestamp
    except (ValueError, TypeError):
        return None, None
    

def get_camera_by_filename(filename, cameras_dict):
    """
    Cerca di identificare la telecamera controllando se uno dei pattern 
    definiti è presente nel nome del file.
    """
    filename_lower = filename.lower()
    
    for cam_id, config in cameras_dict.items():
        patterns = config.get("search_patterns", [])
        # Aggiungiamo di default anche l'ID della camera come pattern
        patterns.append(f"_{cam_id}_") 
        
        for pattern in patterns:
            if pattern.lower() in filename_lower:
                return config
                
    return None


def get_crop_coordinates(bbox, frame_shape, margin_perc=1.0):
   
    x1, y1, x2, y2 = bbox
    h, w = frame_shape[:2]
    
    margin_w = int((x2 - x1) * margin_perc)
    margin_h = int((y2 - y1) * margin_perc)
    
    return [
        max(x1 - margin_w, 0),
        max(y1 - margin_h, 0),
        min(x2 + margin_w, w),
        min(y2 + margin_h, h)
    ]


def get_target_ids(model, settings, mode, camera_ignore_labels):
    """
    Retrieve a list of numeric IDs to monitor based on the selected mode,
    camera‑specific ignore labels, and the set of labels supported by the
    YOLO model.

    Parameters
    ----------
    model : object
        The YOLO model instance used for detection. Its supported
        class labels are extracted to determine which IDs are valid
        for monitoring.
    settings : dict
        Configuration dictionary containing YOLO detection settings,
        including detection groups and other relevant parameters.
    mode : str
        Monitoring mode that dictates which categories of labels to
        consider. Common modes include 'full', 'person', and
        'person_animal'.
    camera_ignore_labels : list[str]
        Labels that should be excluded for a particular camera. These
        are removed from the candidate label list before the final
        target ID list is produced.

    Returns
    -------
    list[int]
        A list of numeric class IDs that should be tracked. The IDs
        are derived from the YOLO model’s supported labels, filtered
        by the selected mode and any camera‑specific ignore labels.

    Notes
    -----
    The method first builds a candidate label list from the active
    detection groups defined in the settings. It then removes any
    labels that the camera has marked to ignore, resulting in the
    final set of labels to monitor. These labels are subsequently
    mapped to numeric IDs using the YOLO model’s class mapping
    [1].
    """
    # 1. Mappa nomi -> ID del modello (es. {'person': 0, 'bicycle': 1...})
    model_name_to_id = {v: k for k, v in model.names.items()}
    
    # 2. Definiamo quali gruppi attivare in base al MODE
    mode_map = {
        "full": ["PERSON", "ANIMAL", "VEHICLE"],
        "person": ["PERSON"],
        "person_animal": ["PERSON", "ANIMAL"]
    }
    active_groups = mode_map.get(mode, ["PERSON"])
    
    # 3. Costruiamo la lista di etichette "candidate" dai gruppi attivi
    candidate_labels = []
    #groups_config = settings["yolo_settings"]["detection_groups"]
    # 1. Recupera yolo_settings (se manca, usa un dizionario vuoto {})
    yolo_conf = settings.get("yolo_settings", {})

    # 2. Recupera detection_groups (se manca, usa una lista o dict vuoto a seconda di cosa ti aspetti)
    groups_config = yolo_conf.get("detection_groups", [])

    for group in active_groups:
        candidate_labels.extend(groups_config.get(group, []))
    
    # 4. Sottraiamo le ignore_labels specifiche della telecamera
    final_labels = [
        label for label in candidate_labels 
        if label not in camera_ignore_labels
    ]
    
    # 5. Trasformiamo i nomi in ID numerici (solo se esistono nel modello)
    target_ids = [
        model_name_to_id[name] 
        for name in final_labels 
        if name in model_name_to_id
    ]
    
    return list(set(target_ids)) # Ritorna ID unici

def calculate_score(category, conf, vision_answer, scoring_settings):
    weights = scoring_settings.get("weights", {})
    multipliers = scoring_settings.get("multipliers", {})

    # 1. Punteggio base basato sulla confidenza YOLO
    if conf >= 0.70:
        base = weights.get("score_high", 3.0)
    elif conf >= 0.55:
        base = weights.get("score_mid", 2.0)
    elif conf >= 0.40:
        base = weights.get("score_low", 1.0)
    else:
        base = 0.3

    # 2. Modificatore basato sulla risposta di Vision
    if vision_answer == category:
        # CONFERMA: Bonus pesante
        multiplier = multipliers.get(category, 1.6)
        base *= multiplier
    
    elif vision_answer == "nothing":
        # SMENTITA: Qui applichiamo il Veto. 
        # Moltiplichiamo per un valore negativo o molto vicino a zero.
        # Se metti -10.0, cancelli istantaneamente ogni accumulo precedente.
        base = -10.0 
            
    else:
        # DISCORDANZA: Vision vede altro (es. YOLO cane, Vision persona)
        # Penalizziamo pesantemente la categoria originale
        base *= 0.1
            
    return base


def cleanup():
    """Funzione dedicata alla pulizia profonda della memoria."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass
    # Se vuoi essere super pulito, puoi aggiungere un log
    log_resource_usage(log, "Free memory")

def get_safe_path(base_dir, camera_name, category, structure_type):
    """
    Costruisce il percorso di destinazione in base alla preferenza dell'utente.
    """
    # Pulizia nome camera (rimuove caratteri non validi per le cartelle)
    safe_camera = re.sub(r'[\\/*?:"<>|]', "", camera_name).strip()
    
    if structure_type == "camera_first":
        return base_dir / safe_camera / category
    elif structure_type == "category_first":
        return base_dir / category / safe_camera
    else: # flat
        return base_dir / category
    
def get_camera_mapping():

    """
    Crea un dizionario semplice { "id": "Nome Umano" } dal file cameras.json.
    """
    CONFIG_DIR = PROJECT_ROOT / "config"
    cameras_config_path = CONFIG_DIR / "cameras.json"
    try:
        data = load_json(cameras_config_path)
        # Se cameras.json ha la struttura {"00": {"name": "Orto", ...}}
        mapping = {cam_id: info.get("name", f"Camera_{cam_id}") 
                   for cam_id, info in data.items()}
        return mapping
    except Exception as e:
        # Fallback in caso di file mancante o corrotto
        return {}
    
def save_test_metrics(output_dir, final_reports, total_time, stats, mode):
    metrics_path = Path(output_dir) / "test_metrics.json"
    
    # Creiamo un riassunto leggibile per capire la velocità dell'hardware
    performance_summary = {}
    for phase, data in stats.items():
        count = data.get("count", 0)
        duration = data.get("time", 0)
        if count > 0:
            avg = duration / count
            performance_summary[phase] = {
                "total_items": count,
                "total_time_sec": round(duration, 2),
                "avg_speed_sec_per_item": round(avg, 3)
            }

    test_report = {
        "session_info": {
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "total_execution_time": round(total_time, 2)
        },
        "performance_summary": performance_summary,
        "detailed_logs": [
            {
                "video": r.get("video_name"),
                "image_filename": Path(r.get("best_frame_path")).name if r.get("best_frame_path") else "N/A",
                "verdict": r.get("category"),
                "engine": r.get("engine"),
                "reasoning": r.get("thinking", "N/A")
            } for r in final_reports
        ]
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_report, f, indent=4, ensure_ascii=False)

def check_dir(dir: str, is_readable=False, is_writeable=False):

    """Controlla se la cartella esiste ed è leggibile o scrivibile.

    Args:
        dir (str): Percorso della cartella da controllare.
        is_readable (bool, optional): Se True, verifica anche se la cartella è leggibile. Defaults to False.
        is_writeable (bool, optional): Se True, verifica anche se la cartella è scrivibile. Defaults to False.

    Returns:
        bool: True se tutte le condizioni sono soddisfatte, False altrimenti.
    """
    import os

    if not os.path.exists(dir):
        return False

    if is_readable and not os.access(dir, os.R_OK):
        return False

    if is_writeable and not os.access(dir, os.W_OK):
        return False

    return True


def validate_ollama_setup(vision_settings):
    """
    Controllo di pre-volo: Ollama è vivo? Il modello qwen3-vl:8b è caricato?
    """
    conf = vision_settings.get('ollama_conf', {})
    ip = conf.get('ip', '127.0.0.1')
    port = conf.get('port', 11434)
    model_required = vision_settings.get('model_name', 'qwen3-vl:8b')
    
    url_tags = f"http://{ip}:{port}/api/tags"
    
    try:
        response = requests.get(url_tags, timeout=3)
        if response.status_code != 200:
            log.info(f"Ollama Server response with error={response.status_code}")
            return False
        
        # Opzionale: controlla se il modello è presente nella lista locale
        models_list = [m['name'] for m in response.json().get('models', [])]
        if model_required not in models_list:
            log.warning(f"Model={model_required} is not reachable. Did you forget to download/activate it?")
            # Non blocchiamo necessariamente, Ollama potrebbe fare il pull automatico, 
            # ma è buono saperlo.
            
        log.info(f"✅ Connect to Ollama ({ip}:{port}). Model={model_required}")
        return True

    except requests.exceptions.ConnectionError:
        log.info(f"Ollama is not reachable on {ip}:{port}.")
        return False
    

def fetch_coords_logic(city_name):
    # LIVELLO 1: PROVA ONLINE (Geopy)
    try:
        # Controllo rapido connessione (opzionale ma consigliato)
        from geopy.geocoders import Nominatim
        geolocator = Nominatim(user_agent="security_app", timeout=3)
        loc = geolocator.geocode(city_name)
        if loc:
            return {"lat": loc.latitude, "lon": loc.longitude}
    except Exception:
        log.critical(f"No internet access. Searchig city='{city_name}' in local db.")

    # LIVELLO 2: FALLBACK OFFLINE (Astral Database)
    try:
        city_data = lookup(city_name, database())
        return {"lat": city_data.latitude, "lon": city_data.longitude}
    except KeyError:
        log.critical(f"City='{city_name}' not found offline. Fallback on Rome")
        return {"lat": 41.89, "lon": 12.49} # Default universale
    

def get_smart_coordinates(city_from_settings):
    update_needed = False

    # # 1. Carichiamo la cache se esiste
    # if COORDS_CACHE_JSON.exists():
    #     try:
    #         with open(COORDS_CACHE_JSON, "r") as f:
    #             cache = json.load(f)
            
    #         if cache.get("city_name") == city_from_settings:
    #             return cache["lat"], cache["lon"]
    #         else:
    #             update_needed = True # Città cambiata nel settings.json
    #     except Exception:
    #         update_needed = True
    # else:
    #     update_needed = True

    # 1. Carichiamo la cache (se il file non esiste o è corrotto, avremo un dizionario vuoto)
    cache = load_json(COORDS_CACHE_JSON) or {}

    # 2. Verifichiamo se la città è la stessa
    # 2. Verifichiamo se la città è la stessa e se le coordinate esistono
    if cache.get("city_name") == city_from_settings:
        lat, lon = cache.get("lat"), cache.get("lon")
        if lat is not None and lon is not None:
            return lat, lon
    
    # 3. Se arriviamo qui (città diversa o dati mancanti), serve aggiornare
    update_needed = True
        

    # 2. Se serve, cerchiamo le nuove coordinate
    # 2. Se serve, cerchiamo le nuove coordinate
    if update_needed:
        coords = fetch_coords_logic(city_from_settings)
        
        # Prepariamo i dati da salvare
        new_cache_data = {
            "city_name": city_from_settings,
            "lat": coords["lat"],
            "lon": coords["lon"]
        }
        
        # Usiamo la tua funzione centralizzata invece di with open
        save_json(new_cache_data, COORDS_CACHE_JSON)
            
        return coords["lat"], coords["lon"]
    

def is_night_astronomic(dt_frame, lat, lon):
    # Observer usa solo matematica locale
    obs = Observer(latitude=lat, longitude=lon)
    
    # Calcolo del sole per la data del video
    s = sun(obs, date=dt_frame.date())
    
    # Applichiamo l'offset di 20 min (crepuscolo civile)
    # Il boost si attiva 20 min prima del tramonto astronomico
    sunset_limit = s["sunset"] - timedelta(minutes=20)
    sunrise_limit = s["sunrise"] + timedelta(minutes=20)
    
    # Portiamo il timestamp del frame in UTC per il confronto
    dt_utc = dt_frame.astimezone(pytz.utc)
    
    return dt_utc > sunset_limit or dt_utc < sunrise_limit