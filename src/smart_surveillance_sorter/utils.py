from datetime import datetime
import json
import logging
import re
import requests
import sys
import cv2
from pathlib import Path
from smart_surveillance_sorter.constants import PROJECT_ROOT
from colorama import Fore, Style
log = logging.getLogger(__name__) 
LABEL_TO_CAT = {
        "person": "PERSON", "people": "PERSON",
        "dog": "ANIMAL", "cat": "ANIMAL", "animal": "ANIMAL",
        "car": "VEHICLE", "truck": "VEHICLE", "vehicle": "VEHICLE"
    }

def load_json(full_path):
 
    if not isinstance(full_path, Path):
        full_path = PROJECT_ROOT / full_path
        
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # Usiamo il logger se disponibile, altrimenti log
        log.critical(f"❌ ERRORE: File non trovato in {full_path}")
        return None
    except json.JSONDecodeError:
        log.critical(f"❌ ERRORE: Il file {full_path} non è un JSON valido.")
        return None

def save_json(data, full_path):
    """
    Salva dati in un file JSON assicurandosi che la cartella esista.
    """
    if not isinstance(full_path, Path):
        full_path = PROJECT_ROOT / full_path
        
    try:
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        log.critical(f"❌ ERRORE durante il salvataggio in {full_path}: {e}")
        return False

def get_video_capture(video_path):
    """Apre un video e restituisce l'oggetto cv2.VideoCapture."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning(f"ERRORE: Impossibile aprire il video {video_path}")
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
    template = storage_settings["filename_template"]
    ts_format = storage_settings["timestamp_format"]

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



# def extract_frames_with_cache(
#     cap,
#     detections,
#     fps,
#     video_path,
#     frames_dir,
#     frames_per_category,
# ):
#     """
#     Extracts frames from a video at specified intervals.
    
#     Parameters
#     ----------
#     video_path : Path
#         Path to the video file to be analyzed.
#     frame_rate : float, optional
#         Number of frames to extract per second (default: 1.0).
#     output_dir : Path | None, optional
#         Directory where the extracted frames will be saved. If `None`, a temporary directory within the input directory is created.
    
#     Returns
#     -------
#     list[Path]
#         List of paths to the extracted frame files.
    
#     Notes
#     -----
#     * The function is used by `Scanner` to prepare data for both `YoloEngine` and `VisionEngine` [5].
#     * Frames are saved in PNG format with sequential naming.
    
#     Examples
#     --------
#     >>> frames = extract_frames(Path("cam01.mp4"), frame_rate=2)
#     >>> len(frames)
#     120
#     """
    
#     saved_frames = []
#     frame_cache = {}
#     frames_dir = Path(frames_dir)
#     # 1. Creiamo una lista piatta di tutti i frame unici che dobbiamo estrarre
#     # Ordiniamo per categoria e confidenza come richiesto
#     target_detections = []
#     for cat, items in detections.items():
#         items.sort(key=lambda x: x["confidence"], reverse=True)
#         # Prendiamo solo i top N per categoria
#         for i, det in enumerate(items[:frames_per_category]):
#             det["cat_rank"] = i
#             det["category"] = cat
#             target_detections.append(det)

#     # 2. Estrazione
#     for det in target_detections:
#         frame_idx = det["frame_idx"]
#         cat = det["category"]
#         rank = det["cat_rank"]

#         # Recupero frame con posizionamento diretto (molto più veloce)
#         if frame_idx not in frame_cache:
#             cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
#             ret, frame = cap.read()
#             if not ret:
#                 continue
#             frame_cache[frame_idx] = frame
#         else:
#             frame = frame_cache[frame_idx]

#         # -------- Calcolo Timestamp --------
#         # Usiamo mtime del file + offset del frame
#         video_start_ts = video_path.stat().st_mtime
#         ts_sec = frame_idx / fps
#         ts_iso = datetime.fromtimestamp(video_start_ts + ts_sec).isoformat()

#         # -------- Salvataggio Frame Intero (PULITO per Vision AI) --------
#         out_name = f"{video_path.stem}_{cat}_{rank}.jpg"
#         out_path = frames_dir / out_name
#         cv2.imwrite(str(out_path), frame)
        

#         # -------- Salva Crop (Dettaglio per controllo) --------
#         # -------- Taglio del Crop --------
#         c_x1, c_y1, c_x2, c_y2 = get_crop_coordinates(det["bbox"], frame.shape)
#         cropped_frame = frame[c_y1:c_y2, c_x1:c_x2]
        
#         out_name_crop = f"{video_path.stem}_{cat}_{rank}_crop.jpg"
#         out_path_crop = frames_dir / out_name_crop
#         if cropped_frame.size > 0: # Evitiamo crash su crop vuoti
#             cv2.imwrite(str(out_path_crop), cropped_frame)
        

#         # -------- Record Unico per Frame + Crop --------
#         saved_frames.append({
#             "category": cat,
#             "frame_path": str(out_path),
#             "crop_path": str(out_path_crop), # <--- Legati insieme
#             "confidence": det["confidence"],
#             "yolo_reliable": det.get("yolo_reliable", False), # Portiamoci dietro il flag per la Vision
#             "bbox": det["bbox"],
#             "area_ratio": det.get("area_ratio", 0),
#             "timestamp": ts_iso,
#             "frame_idx": frame_idx
#         })

#     return saved_frames


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
    groups_config = settings["yolo_settings"]["detection_groups"]
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

# def calculate_score(category, conf, vision_answer, scoring_settings):
#     """
#     Compute a confidence score for a detection by combining YOLO confidence
#     with a Vision model response.

#     Parameters
#     ----------
#     category : str
#         The class name predicted by YOLO for the current detection.
#     conf : float
#         YOLO confidence score for the detection (range 0–1).
#     vision_answer : str
#         The category returned by the Vision model for the same detection.
#         Typical values are the same category name, ``"nothing"``, or other
#         labels defined by the Vision model.
#     scoring_settings : dict
#         A dictionary containing two optional sub‑dictionaries:

#         * ``weights`` – mapping confidence thresholds to base scores
#           (e.g. ``{"score_high": 3.0, "score_mid": 2.0, "score_low": 1.0}``).
#         * ``multipliers`` – mapping categories to a multiplier that is
#           applied when Vision confirms the YOLO category
#           (default multiplier is ``1.6``).

#     Returns
#     -------
#     float
#         The final score for the detection.  The score is calculated as:

#         1. A base score is chosen according to ``conf``:
#            * ``conf`` ≥ 0.70 → ``score_high`` (default 3.0)
#            * 0.55 ≤ ``conf`` < 0.70 → ``score_mid`` (default 2.0)
#            * 0.40 ≤ ``conf`` < 0.55 → ``score_low`` (default 1.0)
#            * ``conf`` < 0.40 → 0.3

#         2. If ``vision_answer`` matches ``category``, the base score is
#            multiplied by the corresponding value in ``multipliers``
#            (default 1.6).

#         3. If ``vision_answer`` is ``"nothing"``, the score remains
#            unchanged (the snippet only shows the multiplier branch,
#            but the logic for ``"nothing"`` can be extended as needed).

#     Notes
#     -----
#     This function is used by the pipeline to weigh detections before
#     aggregating them into an overall alert level.  The exact values for
#     ``score_high``, ``score_mid``, and ``score_low`` as well as the
#     default multiplier are configurable through ``scoring_settings``.
#     The implementation follows the logic outlined in the project’s
#     ``utils.py`` module [1].
#     """
#     weights = scoring_settings.get("weights", {})
#     multipliers = scoring_settings.get("multipliers", {})

#     # 1. Punteggio base basato sulla confidenza YOLO
#     if conf >= 0.70:
#         base = weights.get("score_high", 3.0)
#     elif conf >= 0.55:
#         base = weights.get("score_mid", 2.0)
#     elif conf >= 0.40:
#         base = weights.get("score_low", 1.0)
#     else:
#         base = 0.3

#     # 2. Modificatore basato sulla risposta di Vision
#     if vision_answer == category:
#         # Se Vision conferma esattamente la categoria di YOLO
#         multiplier = multipliers.get(category, 1.6)
#         base *= multiplier
#     elif vision_answer == "nothing":
#         # Se Vision non vede nulla, non diamo bonus ma non penalizziamo troppo
#         base *= 1.0
#     else:
#         # Se Vision vede una categoria DIVERSA (es. YOLO dice animal, Vision dice person)
#         # Penalizziamo la rilevazione corrente
#         base *= 0.5
            
#     return base

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
            log.info(f"❌ Server Ollama risponde con errore {response.status_code}")
            return False
        
        # Opzionale: controlla se il modello è presente nella lista locale
        models_list = [m['name'] for m in response.json().get('models', [])]
        if model_required not in models_list:
            log.info(f"⚠️ Attenzione: il modello {model_required} non risulta tra quelli scaricati.")
            # Non blocchiamo necessariamente, Ollama potrebbe fare il pull automatico, 
            # ma è buono saperlo.
            
        log.info(f"✅ Connessione a Ollama ({ip}:{port}) stabilita. Modello target: {model_required}")
        return True

    except requests.exceptions.ConnectionError:
        log.info(f"❌ Errore Fatale: Ollama è spento su {ip}:{port}.")
        return False