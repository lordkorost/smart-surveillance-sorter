from datetime import datetime, timedelta
import logging
from pathlib import Path
import re
import time

from smart_surveillance_sorter.file_sorter import FileSorter

log = logging.getLogger(__name__)

def build_index(input_dir,settings):
      
        log.info(f"Create index of files in folder={input_dir}...")
        index = {}
        # Recuperiamo le impostazioni dal config
        # Se non esistono, usiamo dei default (Reolink style)
        storage_cfg = settings.get("storage_settings", {})
        template = storage_cfg.get("filename_template", "{nvr_name}_{camera_id}_{timestamp}")
        ts_format = storage_cfg.get("timestamp_format", "%Y%m%d%H%M%S")
        input_dir = Path(input_dir)
        extensions = {".mp4", ".mkv", ".avi", ".mov", ".jpg", ".jpeg"}
        file_list = [f for f in input_dir.iterdir() if f.suffix.lower() in extensions]

        for f in file_list:
            # --- PROTEZIONE FILE IN SCRITTURA ---
            # Se il file è stato modificato molto recentemente (es. negli ultimi 15 secondi)
            if time.time() - f.stat().st_mtime < 15:
                size_pre = f.stat().st_size
                time.sleep(1) # Breve attesa per confermare la scrittura
                size_post = f.stat().st_size
                
                if size_pre != size_post or size_post == 0:
                    # Se la dimensione cambia o è zero, l'NVR ci sta ancora lavorando
                    # Non logghiamo con log.info per non intasare, debug è meglio
                    continue 
            # -------------------------------------
            cam_id_raw, ts = parse_filename(f, template, ts_format)
            
            if cam_id_raw is None or ts is None:
                continue
            
            # Normalizzazione: Qualsiasi cosa esca da parse_filename (es. "0", "1", "01")
            # la trasformiamo nel formato "00", "01" per matchare il tuo cameras.json
            try:
                cam_id = str(cam_id_raw).zfill(2) 
            except:
                cam_id = cam_id_raw

            if cam_id not in index:
                index[cam_id] = []    
            #file_type = "video" if f.suffix.lower() == ".mp4" else "image"
            # --- CORREZIONE QUI ---
            video_extensions = {".mp4", ".mkv", ".avi", ".mov"}
            file_type = "video" if f.suffix.lower() in video_extensions else "image"
            # ----------------------
            index[cam_id].append({
                "type": file_type,
                "timestamp": ts,
                "path": f,
                "cam_id": cam_id
            })

        # Ordinamento e validazione
        for cam_id in index:
            # Ordina per timestamp
            index[cam_id].sort(key=lambda x: x["timestamp"])
            
            # Statistiche veloci
            vids = sum(1 for x in index[cam_id] if x["type"] == "video")
            imgs = sum(1 for x in index[cam_id] if x["type"] == "image")
            log.debug(f"Camera={cam_id}: video={vids} , img={imgs}.")
            
            if imgs < vids:
                log.warning(f"Camera={cam_id} has less images than videos")

        return index
     
def associate_files(index,input_dir):
    
        log.info(f"Starting associations videos-nvr images in folder={input_dir}")
        associations = {}

        for cam_id, timeline in index.items():
            # Dividiamo per tipo (sono già ordinati per timestamp dal _build_index)
            videos = [x for x in timeline if x["type"] == "video"]
            images = [x for x in timeline if x["type"] == "image"]
            
            associations[cam_id] = []

            for i, video in enumerate(videos):
                video_ts = video["timestamp"]
                
                # Calcolo del confine (Boundary)
                MAX_DELTA_SECONDS=40
                upper_bound = video_ts + timedelta(seconds=MAX_DELTA_SECONDS)
                
                # Se c'è un video successivo, restringiamo il confine
                if i + 1 < len(videos):
                    next_video_ts = videos[i+1]["timestamp"]
                    if next_video_ts < upper_bound:
                        upper_bound = next_video_ts

                # Ricerca candidati (immagini non assegnate nel range)
                candidates = [
                    img for img in images
                    if not img.get("assigned", False)
                    and video_ts <= img["timestamp"] < upper_bound
                ]

                # Creazione record di associazione
                assoc_record = {
                    "video_path": video["path"],
                    "video_ts": video_ts,
                    "nvr_images": [],
                    "cam_id": cam_id
                }

                if candidates:
                    # Marciamo le immagini come usate
                    for img in candidates:
                        img["assigned"] = True
                    
                    assoc_record["nvr_images"] = [c["path"] for c in candidates]
                    #log.debug(f"Cam_id={cam_id} Video={video['path'].name} -> {len(candidates)} immagini associate")
                

                associations[cam_id].append(assoc_record)

        return associations

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
    
def sortVideos(settings,
               input_dir,
               output_dir,
               is_test,
               final_data,
               results,
               full_index):
                
    # Inizializzazione e avvio Sorter
    file_sorter = FileSorter(settings,
                             input_dir, 
                             output_dir, 
                             is_test)
                
    #Il Sorter riceve il verdetto, i dati grezzi (per i frame) e l'indice (per NVR/Others)
    file_sorter.sort_all(final_data, 
                         results, 
                         full_index)