from datetime import datetime
import json
import logging
import re
import requests
import cv2
from pathlib import Path
from smart_surveillance_sorter.constants import PROJECT_ROOT
import json
from astral.geocoder import database, lookup
from smart_surveillance_sorter.constants import COORDS_CACHE_JSON
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
    """Load JSON data from file.
    
    Args:
        full_path: Path to JSON file (Path object or string)
        
    Returns:
        Loaded JSON data or None if file not found or invalid JSON
    """
    if not isinstance(full_path, Path):
        full_path = PROJECT_ROOT / full_path
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        log.critical(f"Not found file={full_path}")
        return None
    except json.JSONDecodeError:
        log.critical(f"File={full_path} is not a valid JSON.")
        return None

def save_json(data, full_path):
    """Save data to JSON file, creating parent directories as needed.
    
    Args:
        data: Data to save (will be converted to JSON)
        full_path: Path where to save the JSON file (Path object or string)
        
    Returns:
        True if successful, False otherwise
    """
    if not isinstance(full_path, Path):
        full_path = PROJECT_ROOT / full_path
        
    try:
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        log.critical(f"Error during save on path={full_path}. err={e}")
        return False

def get_video_capture(video_path):
    """Open a video file and return cv2.VideoCapture object.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        cv2.VideoCapture object if successful, None otherwise
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning(f"Error is not possible open vid={video_path}")
        return None
    return cap

def get_camera_by_filename(filename, cameras_dict):
    """Identify which camera a file belongs to by examining search patterns.
    
    Attempts to identify the camera by checking if one of the defined
    search patterns is present in the filename.
    
    Args:
        filename: Name of the file to check
        cameras_dict: Dictionary of camera configurations
        
    Returns:
        Camera configuration dict if match found, None otherwise
    """
    filename_lower = filename.lower()
    
    for cam_id, config in cameras_dict.items():
        patterns = config.get("search_patterns", [])
        # Also add the camera ID as a default pattern
        patterns.append(f"_{cam_id}_") 
        
        for pattern in patterns:
            if pattern.lower() in filename_lower:
                return config
                
    return None

def get_crop_coordinates(bbox, frame_shape, margin_perc=1.0):
    """Calculate crop coordinates with margin from a bounding box.
    
    Args:
        bbox: Bounding box as [x1, y1, x2, y2]
        frame_shape: Shape of the frame (height, width, ...)
        margin_perc: Margin percentage relative to bbox size (default: 1.0 = 100%)
        
    Returns:
        List of cropped coordinates [x1, y1, x2, y2] clamped to frame boundaries
    """
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
    # 1. Model name -> ID mapping (e.g. {'person': 0, 'bicycle': 1...})
    model_name_to_id = {v: k for k, v in model.names.items()}
    
    # 2. Define which groups to activate based on MODE
    mode_map = {
        "full": ["PERSON", "ANIMAL", "VEHICLE"],
        "person": ["PERSON"],
        "person_animal": ["PERSON", "ANIMAL"]
    }
    active_groups = mode_map.get(mode, ["PERSON"])
    
    # 3. Build list of "candidate" labels from active groups
    candidate_labels = []
    #groups_config = settings["yolo_settings"]["detection_groups"]
    # 1. Retrieve yolo_settings (if missing, use empty dict {})
    yolo_conf = settings.get("yolo_settings", {})

    # 2. Retrieve detection_groups (if missing, use empty list/dict)
    groups_config = yolo_conf.get("detection_groups", [])

    for group in active_groups:
        candidate_labels.extend(groups_config.get(group, []))
    
    # 4. Remove camera-specific ignore labels
    final_labels = [
        label for label in candidate_labels 
        if label not in camera_ignore_labels
    ]
    
    # 5. Transform names to numeric IDs (only if they exist in the model)
    target_ids = [
        model_name_to_id[name] 
        for name in final_labels 
        if name in model_name_to_id
    ]
    
    return list(set(target_ids)) # Return unique IDs

def calculate_score(category, conf, vision_answer, scoring_settings):
    """Calculate detection score based on confidence and vision model agreement.
    
    Combines YOLO confidence with vision model verification to produce a final score.
    Applies multipliers and weights based on configuration.
    
    Args:
        category: Detected object category
        conf: YOLO confidence score (0-1)
        vision_answer: Vision model's assessment of the detection
        scoring_settings: Dict with "weights" and "multipliers" configuration
        
    Returns:
        Final score after applying weights and multipliers
    """
    weights = scoring_settings.get("weights", {})
    multipliers = scoring_settings.get("multipliers", {})

    # 1. Base score calculation based on YOLO confidence
    if conf >= 0.70:
        base = weights.get("score_high", 3.0)
    elif conf >= 0.55:
        base = weights.get("score_mid", 2.0)
    elif conf >= 0.40:
        base = weights.get("score_low", 1.0)
    else:
        base = 0.3

    # 2. Modifier based on Vision model response
    if vision_answer == category:
        # CONFIRMATION: Heavy bonus
        multiplier = multipliers.get(category, 1.6)
        base *= multiplier
    
    elif vision_answer == "nothing":
        base = -10.0 
            
    else:
        base *= 0.1
            
    return base

def cleanup():
    """Clean up memory by running garbage collection and clearing GPU cache."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass
   
    log_resource_usage(log, "Free memory")

def get_safe_path(base_dir, camera_name, category, structure_type):
    """Build output file path based on user's preferred folder structure.
    
    Args:
        base_dir: Base output directory
        camera_name: Name of the camera
        category: Content category
        structure_type: Either "camera_first", "category_first", or "flat"
        
    Returns:
        Constructed Path object for the destination directory
    """
    safe_camera = re.sub(r'[\\/*?:"<>|]', "", camera_name).strip()
    
    if structure_type == "camera_first":
        return base_dir / safe_camera / category
    elif structure_type == "category_first":
        return base_dir / category / safe_camera
    else: # flat
        return base_dir / category
    
def get_camera_mapping():

    """Create a simple { "id": "Human Name" } mapping from cameras.json.
    
    Returns:
        Dictionary mapping camera IDs to camera names, empty dict if error
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
        return {}
    
def save_test_metrics(output_dir, final_reports, total_time, stats, mode,settings=None):
    """Save test execution metrics and performance statistics to JSON file.
    
    Args:
        output_dir: Directory where metrics file will be saved
        final_reports: List of detection reports
        total_time: Total execution time in seconds
        stats: Dictionary with performance statistics per phase
        mode: Detection mode used
        settings: Optional settings dict for parameter logging
    """
    metrics_path = Path(output_dir) / "test_metrics.json"
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


    # Extract relevant parameters from settings
    yolo_params = {}
    engine_params = {}
    if settings:
        ys = settings.get("yolo_settings", {})
        dss = ys.get("dynamic_stride_settings", {})
        yolo_params = {
            "model": ys.get("model_path", ""),
            "device": ys.get("device", ""),
            "vid_stride_sec": ys.get("vid_stride_sec", 0.6),
            "num_occurrence": ys.get("num_occurrence", 3),
            "time_gap_sec": ys.get("time_gap_sec", 3),
            "warmup_sec": dss.get("warmup_sec", 5),
            "stride_fast_sec": dss.get("stride_fast_sec", 1.0),
            "pre_roll_sec": dss.get("pre_roll_sec", 20),
        }
        vs = settings.get("vision_settings", {})
        engine_params = {
            "vision_model": vs.get("model_name", ""),
            "temperature": vs.get("temperature", 1),
        }

    test_report = {
        "session_info": {
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "total_execution_time": round(total_time, 2)
        },
        "yolo_params": yolo_params,
        "engine_params": engine_params,
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

    """Check if a directory exists and verify read/write permissions.

    Args:
        dir (str): Path to the directory to check.
        is_readable (bool, optional): If True, also verify read access. Defaults to False.
        is_writeable (bool, optional): If True, also verify write access. Defaults to False.

    Returns:
        bool: True if all conditions are met, False otherwise.
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
    """Pre-flight check: is Ollama server active and is the required model loaded?
    
    Args:
        vision_settings: Vision settings dict containing ollama_conf and model_name
        
    Returns:
        True if Ollama server is reachable and model is available, False otherwise
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
        
        models_list = [m['name'] for m in response.json().get('models', [])]
        if model_required not in models_list:
            log.warning(f"Model={model_required} is not reachable. Did you forget to download/activate it?")
            
        log.info(f"✅ Connect to Ollama ({ip}:{port}). Model={model_required}")
        return True

    except requests.exceptions.ConnectionError:
        log.info(f"Ollama is not reachable on {ip}:{port}.")
        return False
    
def fetch_coords_logic(city_name):
    """Fetch geocoordinates for a city using online and offline methods.
    
    Attempts to fetch coordinates online first (Geopy), falls back to offline
    Astral database if no internet, and defaults to Rome if city not found.
    
    Args:
        city_name: Name of the city to find coordinates for
        
    Returns:
        Dictionary with "lat" and "lon" keys
    """
    # LEVEL 1: Try online (Geopy)
    try:
        # Controllo rapido connessione
        from geopy.geocoders import Nominatim
        geolocator = Nominatim(user_agent="security_app", timeout=3)
        loc = geolocator.geocode(city_name)
        if loc:
            return {"lat": loc.latitude, "lon": loc.longitude}
    except Exception:
        log.critical(f"No internet access. Searching city='{city_name}' in local db.")

    # LEVEL 2: OFFLINE fallback (Astral Database)
    try:
        city_data = lookup(city_name, database())
        return {"lat": city_data.latitude, "lon": city_data.longitude}
    except KeyError:
        log.critical(f"City='{city_name}' not found offline. Fallback on Rome")
        return {"lat": 41.89, "lon": 12.49} # Default universal fallback    

def get_smart_coordinates(city_from_settings):
    """Get and cache geocoordinates for the configured city.
    
    Checks if coordinates are cached for the current city, returns cached values if valid,
    otherwise fetches new coordinates and updates cache.
    
    Args:
        city_from_settings: City name from configuration
        
    Returns:
        Tuple of (latitude, longitude)
    """
    update_needed = False

    # 1. Load cache (empty dict if file missing or corrupted)
    cache = load_json(COORDS_CACHE_JSON) or {}

    # 2. Check if city is the same and coordinates exist
    if cache.get("city_name") == city_from_settings:
        lat, lon = cache.get("lat"), cache.get("lon")
        if lat is not None and lon is not None:
            return lat, lon
    
    # 3. If we reach here (different city or missing data), we need to update
    update_needed = True

    # 2. If needed, fetch new coordinates
    if update_needed:
        coords = fetch_coords_logic(city_from_settings)
        
        # Prepare data to save
        new_cache_data = {
            "city_name": city_from_settings,
            "lat": coords["lat"],
            "lon": coords["lon"]
        }
        
        save_json(new_cache_data, COORDS_CACHE_JSON)
            
        return coords["lat"], coords["lon"]  

def is_night_astronomic(dt_frame, lat, lon):
    """Check if a given time is within astronomical night for a location.
    
    Determines if the frame timestamp is in astronomical night, applying a
    20-minute offset before sunset for civil twilight detection.
    
    Args:
        dt_frame: Datetime of the frame (with timezone info)
        lat: Latitude of the location
        lon: Longitude of the location
        
    Returns:
        True if the frame is during night, False otherwise
    """
    # Observer uses only local mathematics
    obs = Observer(latitude=lat, longitude=lon)
    
    # Calculate sun position for the video date
    s = sun(obs, date=dt_frame.date())
    
    # Apply 20 min offset (civil twilight)
    # Detection boost activates 20 min before astronomical sunset
    sunset_limit = s["sunset"] - timedelta(minutes=20)
    sunrise_limit = s["sunrise"] + timedelta(minutes=20)
    
    # Convert frame timestamp to UTC for comparison
    dt_utc = dt_frame.astimezone(pytz.utc)
    
    return dt_utc > sunset_limit or dt_utc < sunrise_limit