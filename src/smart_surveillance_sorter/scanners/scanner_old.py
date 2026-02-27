from datetime import timedelta
#import json
from pathlib import Path
import sys
import time

import torch
from smart_surveillance_sorter.constants import CAMERAS_JSON, CHECKS_DIR, CLIPBLIP_CACHE, FINAL_REPORT, FRAME_DIR, LENS_HEALTH, PROJECT_ROOT, SETTINGS_JSON, VISION_CACHE, YOLO_CACHE
import logging
from pathlib import Path
from smart_surveillance_sorter.file_sorter import FileSorter
from smart_surveillance_sorter.logger import detect_device, get_pbar_prefix, log_device_status, log_resource_usage
from smart_surveillance_sorter.scanners.clip_blip_engine import ClipBlipEngine
from smart_surveillance_sorter.scanners.yolo_engine import YoloEngine
from smart_surveillance_sorter.utils import load_json, parse_filename, save_json, save_test_metrics, validate_ollama_setup
from tqdm import tqdm
from colorama import Fore, Style



log = logging.getLogger(__name__)

class Scanner():
    """
    Scanner base class for all surveillance folder scanners.

    The :class:`Scanner` orchestrates the end‑to‑end process of
    analysing a directory of video files.  It centralises the
    configuration of the underlying YOLO detection engine,
    handles logging, and coordinates the indexing, YOLO
    inference, optional Vision‑AI refinement, and fallback
    analysis.

    Parameters
    ----------
    mode : str
        Operational mode for the YOLO engine (e.g., "fast", "full").
        Determines how the underlying model processes input data.
    device : str, optional
        Target inference device such as "cpu" or "cuda:0".  If omitted,
        the YoloEngine defaults to its configured device.
    is_refine : bool, default False
        When ``True`` the best YOLO detections are sent to a Vision
        AI backend (e.g., Ollama/Qwen) for further refinement.
    is_fallback : bool, default False
        When ``True`` a secondary scan is performed for any events
        that were not fully processed by the primary YOLO pass.

    Attributes
    ----------
    mode : str
        The mode passed to the constructor.
    device : str or None
        The device passed to the constructor.
    is_refine : bool
        Indicates whether Vision‑AI refinement is enabled.
    is_fallback : bool
        Indicates whether a fallback scan should be performed.
    logger : logging.Logger
        Logger instance used throughout the class.
    yolo_engine : YoloEngine
        Instance of the YOLO engine configured with the selected
        mode, device, and logger.

    Notes
    -----
    The class does not perform any path calculations; constants such
    as :data:`CAMERAS_JSON` and :data:`TEMP_DIR` are imported from
    a centralised configuration module.  Upon initialization it
    logs a message summarising the configuration [1].
    """
    def __init__(self, mode, device=None, is_refine=False, is_fallback=False, is_test = False,engine="blip",is_check_clean=False,is_real_time=False):
       # 1. Inizializzazione fuori dal loop
        #self.monitor = ResourceMonitorAMD()
        
        # 1. Parametri operativi
        self.mode = mode
        self.is_refine = is_refine
        self.is_fallback = is_fallback
        self.is_test = is_test
        self.engine = engine
        self.is_check_clean = is_check_clean
        self.is_real_time = is_real_time
   

        if self.is_test:
            self.stats = {
            "yolo_images": {"count": 0, "time": 0},
            "yolo_videos": {"count": 0, "time": 0},
            "vision_refine": {"count": 0, "time": 0},
            "vision_fallback": {"count": 0, "time": 0}
            }

        self.settings              =  load_json(SETTINGS_JSON)
        self.cameras_config        =  load_json(CAMERAS_JSON)

        if self.engine == "vision":
            self.vision_cfg = self.settings.get("vision_settings", {})
            # Qui usiamo il valore di ritorno (True/False) per decidere se continuare
            if not validate_ollama_setup(self.vision_cfg):
                sys.exit(1) # Questo blocca tutto e ti riporta al terminale

        yolo_cfg = self.settings.get("yolo_settings", {})
        
        if device is not None:
            self.device = device
        else:
            # self.device = self.settings.get("device", "cuda" if torch.cuda.is_available() else "cpu")
            # Cerca dentro yolo_settings, e se non c'è usa 'cuda' come fallback
            self.device = yolo_cfg.get("device", "cpu")

        #log.info(f"🛠️ [Scanner] Inizializzato su device={self.device}")
        #device_str, torch_dev = detect_device()
        # 2. Passa le variabili ottenute, non la funzione
        #log_device_status(log, device_str, torch_dev)
        log_resource_usage(log, "START")
        # 3. Stato dell'analisi
        self.results = []               # Risultati pronti per il report/sorting
        self.resolved_videos = set()    # Per evitare di ri-analizzare video già risolti
        self.frames_dir = None          # Verrà impostato dinamicamente durante lo scan

        self.vision_engine = None # Caricato on-demand da _ensure_vision_initialized

        # Istanza del motore 
        self.yolo_engine = YoloEngine(
            mode=self.mode,
            device=self.device,
            settings=self.settings,
            cameras_config=self.cameras_config
            
        )
        
        # log.info(
        #     f"🚀 {self.__class__.__name__} inizializzato | "
        #     f"Mode={mode} | Refine={is_refine} | Device={self.device}"
        # )
    
   
    def scan_folder(self, input_dir,output_dir):
        """
        Esegue il flusso completo: Scansione -> Analisi (YOLO/Vision) -> Finalizzazione -> Sorting.
        """
        # --- TIMER INIZIO ---
        t_start = time.time()
        #self.monitor.log_stats("AVVIO SISTEMA")
        # 1. Normalizzazione Input
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # 2. Setup Directory di Lavoro
        self.frames_dir = self.output_dir / FRAME_DIR
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Frames directory={self.frames_dir}")

        # 3. Indicizzazione e Associazione
        raw_index = self._build_index(self.input_dir)
        self.full_index = self._associate_files(raw_index)

        yolo_cache_file = Path(self.output_dir) / YOLO_CACHE
        self.yolo_cache_file = yolo_cache_file

        log.debug(f"DEBUG: full_index type={type(self.full_index)}")

        if self.full_index:
            first_cam = list(self.full_index.keys())[0]
            videos_della_cam = self.full_index[first_cam]
            
            if videos_della_cam: # 
                first_video = videos_della_cam[0]
                log.debug(f"First video in full_index: vid={first_video}")
            else:
                log.debug(f"Camera cam={first_cam} no video.")
        if yolo_cache_file.exists():
            log.warning("YOLO Cache found. Resume prev results.")
            try:
                loaded_data = load_json(yolo_cache_file)
                # Assicuriamoci che sia una lista di dizionari
                self.results = [r for r in loaded_data if isinstance(r, dict)]
            except Exception:
                log.error("YOLO Cache corrupted. Resume abort, starting new scan.")
                self.results = []

            for r in self.results:
                if "video_path" in r:
                    self.resolved_videos.add(str(r["video_path"]))
        else:
            self.results = []

        # 3. FILTRAGGIO INDICE
        original_video_count = sum(len(v_list) for v_list in self.full_index.values())
        
        processed_set = {str(r.get("video_path")) for r in self.results if isinstance(r, dict) and r.get("video_path")}

        # Filtriamo mantenendo la struttura
        self.full_index = {
            cam_id: [
                v for v in v_list 
                if str(v.get("video_path")) not in processed_set
            ]
            for cam_id, v_list in self.full_index.items()
        }

        # Ricalcoliamo il totale reale dei video rimasti
        new_video_count = sum(len(v_list) for v_list in self.full_index.values())
        skipped = original_video_count - new_video_count

        if skipped > 0:
            log.info(f"Skipped video={skipped} in cache. New video to process {new_video_count} videos")
        
        # 4. Avvio Scansione basato sui VIDEO, non sulle chiavi del dizionario
        if new_video_count > 0:
            if not processed_set:
                log.info("NVR Images scan for person.")
                self._scan_images(self.full_index)
            
            log.info(f"YOLO scan on new videos={new_video_count}")
            log_resource_usage(log, "YOLO")
            self._scan_videos(self.full_index)
        else:
            log.info("All video processed.")
        
    
        # Lo scopo qui è popolare 'final_data', l'unico oggetto che il Sorter leggerà.
        final_data = []
        self.final_reports = []
        if self.is_refine and self.engine == "blip":
            log_resource_usage(log, "BLIP")
            self._clip_blip_scan_refine()

            if self.is_fallback:
               
                vision_queue = self._get_arbitration_queue()
                if vision_queue:
                     # ... logica Vision AI ...
                    log.info("Fallback step with vision on blip res")
                    log_resource_usage(log, "VISION")
                    self._ensure_vision_initialized() 
                    log.info(f"{len(vision_queue)} to process.")
                    # Passiamo la coda filtrata al motore Vision
                    self._fallback_with_vision(vision_queue)

            log.info("Scan end. Start folder sorting.")
            self.final_reports = self.clip_blip_results
            final_data = self._finalize_results(self.final_reports, engine="blip")
            log_resource_usage(log, "END")
            
        elif self.is_refine and self.engine == "vision":
            
            # ... logica Vision AI ...
            log.info("Starting refine with vision")
            
            self._ensure_vision_initialized() 
            
            if self.is_check_clean:
                log_resource_usage(log, "VISION")
                self.lens_status = self.check_cameras_clean()
            
                # Percorso del file JSON di report
                health_report_path = Path(self.output_dir) / LENS_HEALTH
                
                save_json(self.lens_status, health_report_path)
                log.info(f"Cameras lens status report saved in folder={health_report_path}")
          
            # self._refine_with_vision popola internamente self.final_reports
            log_resource_usage(log, "VISION")
            self._refine_with_vision() 
            final_data = self._finalize_results(self.final_reports, engine="vision")
            log_resource_usage(log, "END VISION")
         
           

        # 6. Fase 4: Sorter (Il Gran Finale)
        if final_data:
            # Salvataggio unico del verdetto
            final_report_path = self.output_dir / FINAL_REPORT
            save_json(final_data, final_report_path)
            log.info(f"Final results saved in file={final_report_path}")
            
            # Inizializzazione e avvio Sorter
            self.file_sorter = FileSorter(
                self.settings, 
                self.input_dir, 
                self.output_dir, 
                self.is_test
            )
            
            #Il Sorter riceve il verdetto, i dati grezzi (per i frame) e l'indice (per NVR/Others)
            # self.file_sorter.sort_all(
            #     final_results=final_data, 
            #     raw_results=self.results, 
            #     full_index=self.full_index
            # )

            #self.file_sorter.cleanup()

            total_time = time.time()-t_start
            if(self.is_test):
                save_test_metrics(output_dir, self.final_reports, total_time, self.stats, self.mode)
                

    def _build_index(self, input_dir):
      
        log.info(f"Create index of files in folder={input_dir}...")
        index = {}
        
        # Recuperiamo le impostazioni dal config
        # Se non esistono, usiamo dei default (Reolink style)
        storage_cfg = self.settings.get("storage_settings", {})
        template = storage_cfg.get("filename_template", "{nvr_name}_{camera_id}_{timestamp}")
        ts_format = storage_cfg.get("timestamp_format", "%Y%m%d%H%M%S")

        #extensions = {".mp4", ".jpg", ".jpeg"}
        extensions = {".mp4", ".mkv", ".avi", ".mov", ".jpg", ".jpeg"}
        file_list = [f for f in input_dir.iterdir() if f.suffix.lower() in extensions]

        import time # Assicurati che sia importato in cima al file o qui

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
    
    def _associate_files(self, index):
        """
        Associate video files with NVR images for each camera.

        This method takes an index mapping camera IDs to lists of media
        items and creates associations between video files and the
        corresponding image snapshots that belong to the same time window.
        It handles temporal boundaries to avoid overlapping associations
        and logs statistics for each camera.

        Parameters
        ----------
        index : dict
            Mapping of camera identifiers to a list of media dictionaries.
            Each dictionary contains at least the keys ``type`` (either
            ``"video"`` or ``"image"``) and the media file path.

        Returns
        -------
        dict
            A dictionary where keys are camera IDs and values are lists
            of association tuples or dictionaries linking a video to its
            corresponding images.

        Notes
        -----
        The implementation follows the logic described in the original
        scanner module. It logs progress and warnings about potential
        imbalances between video and image counts per camera.
        """
        log.info(f"Starting associations videos-nvr images in folder={self.input_dir}")
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
    
    def _scan_images(self, associations):
       
        # --- TIMER INIZIO ---
        t_start = time.time()
     
        tasks = []
        for cam_id, items in associations.items():
            for item in items:
                video_path = str(item["video_path"])
                nvr_images = item.get("nvr_images", [])

                # Filtro di inclusione per la lista tasks
                if nvr_images and video_path not in self.resolved_videos:
                    tasks.append((cam_id, item, nvr_images))
                # if not nvr_images or video_path in self.resolved_videos:
                #     continue
        # Se non ci sono immagini NVR da analizzare, usciamo subito
        if not tasks:
            log.info("No nvr images to scan")
            return

        # 1. INFO Iniziale
        msg_start = f"Yolo scan on num_img={len(tasks)}"
        log.info(msg_start) # Logga su file
        #tqdm.write(f"[{time.strftime('%H:%M:%S')}] - ℹ️ {msg_start}") # Stampa pulito a schermo
        
        # Avvio Progress Bar
        pbar = tqdm(
            tasks,
            desc="Progress",
            unit="it",
            ncols=100,
            bar_format=f"{get_pbar_prefix('YOLO Scan')} {{rate_fmt:>10}} [{{bar}}] {{percentage:3.0f}}% {{n_fmt}}/{{total_fmt}} {{elapsed}}<{{remaining}}"
        )
        
        count_resolved = 0
        for cam_id, item, nvr_images in pbar:
            video_path = str(item["video_path"])
            
            # Qui dentro fai il lavoro vero
            for img_path in nvr_images:
                result = self.yolo_engine.scan_single_image(img_path, item["video_path"], self.frames_dir, cam_id)
                if result:
                    self.results.append(result)
                    self.resolved_videos.add(video_path)
                    count_resolved += 1
                    break 
        
        # 3. Chiudi alla fine
        pbar.close()
        # --- SALVATAGGIO STATS ---
        if self.is_test:
            self.stats["yolo_images"]["count"] = len(tasks)
            self.stats["yolo_images"]["time"] = time.time() - t_start

      

    def _scan_videos(self, associations):
        """
        Process unresolved videos by delegating to the appropriate analyzer.

        After the quick image‑based resolution step, any remaining videos
        (i.e., those not present in ``self.resolved_videos``) are handled
        by this method. It walks through the *associations* dictionary,
        converting each ``Path`` object to a string for the analyzer
        and invoking the video‑specific scanning routine. The method also
        logs basic statistics per camera.

        Parameters
        ----------
        associations : dict
            Mapping of camera identifiers to a list of association records.
            Each record must contain the key ``"video_path"`` (a :class:`Path`
            object). The method may use additional keys depending on the
            analyzer implementation.

        Returns
        -------
        None
            Results are accumulated in ``self.results`` and resolved videos
            are tracked in ``self.resolved_videos``; no explicit return value
            is provided.

        Notes
        -----
        By iterating only over videos that remain unresolved after the
        image‑based scan, this function ensures that each video is analyzed
        exactly once. The logic follows the implementation shown in the
        scanner module. [1]
        """
        # --- TIMER INIZIO ---
        t_start = time.time()
        # 1. LOG INFO INIZIALE
        model_name = self.settings.get("yolo_settings").get("model_path")
        log.info(
            f"Process yolo scan videos model={model_name},device={self.device}")

        # 2. Prepariamo i compiti (tasks) per la barra
        tasks = []
        for cam_id, items in associations.items():
            for item in items:
                video_path = item["video_path"] # Path object
                video_str = str(video_path)

                # Includiamo solo se non risolto e se il file esiste
                if video_str not in self.resolved_videos and video_path.exists():
                    tasks.append((cam_id, item, video_path, video_str))
             
        if not tasks:
            log.warning("No videos to scan.")
            return    

        # 1. INFO Iniziale
        msg_start = f"Yolo scan on num_vid={len(tasks)}{Style.RESET_ALL}"
        log.info(msg_start) # Logga su file
        #tqdm.write(f"[{time.strftime('%H:%M:%S')}] - ℹ️ {msg_start}") # Stampa pulito a schermo
        # Invece di tqdm.write manuale, puoi fare così:
        
        # 3. Avvio Progress Bar
        # Progress 1,59s/it [barra] 100% num/tot
        pbar = tqdm(
            tasks,
            desc="Progress",
            unit="it",
            ncols=100,
            bar_format="{desc}: {rate_fmt} [{bar}] {percentage:3.0f}% {n_fmt}/{total_fmt} {elapsed}<{remaining}"
         )
        pbar.write(f"[{time.strftime('%H:%M:%S')}] - ℹ️ {msg_start}")
        for cam_id, item, video_path, video_str in pbar:       
            # Chiamata al metodo che implementeremo nello scanner specifico
            result = self.yolo_engine.scan_video(
            video_path=video_path,
            frames_dir=self.frames_dir,
            cam_id=cam_id
            )

            if result:
                self.results.append(result)
                self.resolved_videos.add(video_str)
                save_json(self.results, self.yolo_cache_file)
        pbar.close()
        # --- SALVATAGGIO STATS ---
        if self.is_test:
            self.stats["yolo_videos"]["count"] = len(tasks)
            self.stats["yolo_videos"]["time"] = time.time() - t_start
    
        # 4. Conteggio finale (opzionale, ma utile)
        log.info(f"All videos processed.")
    
    def _refine_with_vision(self):
        if not self.results:
            log.warning("No YOLO result to refine.")
            return
        
        vision_cache_file = Path(self.output_dir) / VISION_CACHE
        
        # Inizializziamo/Carichiamo i risultati
        if vision_cache_file.exists():
            log.info("Vision AI Cache found! Resume...")
            self.vision_results = load_json(vision_cache_file)
        else:
            self.vision_results = []

        # Creiamo il set per saltare i video già fatti
        processed_vision_paths = {str(r["video_path"]) for r in self.vision_results}
        
        # --- TIMER INIZIO ---
        t_start = time.time()
        # 1. INFO iniziale con modello (come richiesto)
        #log.info(f"Processo Vision refine")

        # 2. Prepariamo i report finali
        self.final_reports = []
        # 1. INFO Iniziale
        msg_start = f"Start refine on num_vid={len(self.results)}"
        log.info(msg_start) # Logga su file
        #tqdm.write(f"[{time.strftime('%H:%M:%S')}] - ℹ️ {msg_start}") # Stampa pulito a schermo
        # Invece di tqdm.write manuale, puoi fare così:
        
        # 3. Avvio Progress Bar
        # Usiamo self.results come iteratore per la barra
        pbar = tqdm(
            self.results,
            desc="Progress",
            unit="it",
            ncols=100,
            bar_format="{desc}: {rate_fmt} [{bar}] {percentage:3.0f}% {n_fmt}/{total_fmt} {elapsed}<{remaining}"
        )
        #pbar.write(f"[{time.strftime('%H:%M:%S')}] - ℹ️ {msg_start}")
        
        for video_data in pbar:
            video_path = str(video_data.get("video_path"))

            # --- 1. SALTO SE GIÀ IN CACHE ---
            if video_path in processed_vision_paths:
                cache_entry = next((r for r in self.vision_results if str(r["video_path"]) == video_path), None)
                if cache_entry:
                    self.final_reports.append(cache_entry)
                pbar.update(1)
                continue

            # --- 2. GESTIONE OTHERS (YOLO DISCARD) ---
            if "others" in video_data.get("categories_found", []):
                cam_id = video_data.get("camera_id")
                cam_info = self.cameras_config.get(str(cam_id), {})
                cam_name = cam_info.get("name", f"Camera_{cam_id}")
                
                report = {
                    "camera_id": str(cam_id),
                    "camera_name": cam_name,
                    "video_name": Path(video_path).name,
                    "video_path": video_path,
                    "category": "others",
                    "confidence": 0,
                    "best_frame_path": None,
                    "engine": "yolo_discard", 
                    "thinking": "No objects detected by YOLO engine."
                }
                self.vision_results.append(report)
                # Salvataggio immediato anche per gli scarti
                save_json(self.vision_results, vision_cache_file)
                pbar.update(1)
                continue 

            # --- 3. CHIAMATA AL MOTORE VISION (QWEN) ---
            report = self.vision_engine.refine_single_video(video_data)
            
            if report:
                self.vision_results.append(report)
                # SALVATAGGIO INCREMENTALE (Vitale per Vision AI)
                save_json(self.vision_results, vision_cache_file)

                # --- LOGGING TEST ---
                if self.is_test and report.get("thinking"):
                    tqdm.write(f"\n{Fore.CYAN}{'='*40}{Style.RESET_ALL}")
                    tqdm.write(f"🧠 {Fore.MAGENTA}REASONING for {report['video_name']}:{Style.RESET_ALL}")
                    tqdm.write(f"{Fore.LIGHTBLACK_EX}{report['thinking']}{Style.RESET_ALL}")
                    tqdm.write(f"🎯 {Fore.GREEN}FINAL VERDICT: {report['category']}{Style.RESET_ALL}")
                    tqdm.write(f"{Fore.CYAN}{'='*40}{Style.RESET_ALL}\n")
            
            pbar.update(1)
        self.final_reports = self.vision_results
        # --- SALVATAGGIO STATS ----
        if self.is_test:
            self.stats["vision_refine"]["count"] = len(self.results)
            self.stats["vision_refine"]["time"] = time.time() - t_start

    def _ensure_vision_initialized(self):
        """
        Lazily initialise the Vision‑AI engine (e.g., Ollama/Qwen) only when required.

        This method follows a *lazy‑loading* pattern to minimise memory usage when
        the Vision mode is not active.  If the vision engine has already been
        created, the method simply returns.  Otherwise it attempts to import
        and instantiate :class:`VisionEngine`, passing the YOLO settings,
        camera configuration, and the current scanning mode.  Failure to
        import the Vision module disables further refinement and logs an
        error.

        The engine is stored in ``self.vision_engine`` and a confirmation
        message is logged upon successful initialisation.  The behaviour
        matches the implementation in the scanner module [1].
        """
        if self.vision_engine is not None:
            return

        log.info("Setting vision Ai (Ollama)")
        
        try:
            from smart_surveillance_sorter.scanners.vision_engine import VisionEngine
            
            self.vision_engine = VisionEngine(
                settings=self.settings,
                cameras_config=self.cameras_config,
                mode=self.mode
                #logger=log
            )

            log.info("Vision AI ready.")
            
        except ImportError as e:
            log.error(f"Error on setting up vision_engine, error={e}")
            self.is_refine = False

   
    def _fallback_scan(self):

        # --- TIMER INIZIO ---
        t_start = time.time()
        identified_video_names = {res['video_name'] for res in self.final_reports}
        
        # 1. INFO Iniziale
        log.info(f"Avvio fallback scan (low-confidence recovery)... ")
        
        video_to_suspects = {} 
        
        # Prepariamo la lista dei video da analizzare per la barra
        pending_videos = []
        for cam_id, records in self.full_index.items():
            for record in records:
                if record["video_path"].name not in identified_video_names:
                    pending_videos.append((cam_id, record))

        if not pending_videos:
            log.info("Nessun video da recuperare nel fallback.")
            return
        # 1. INFO Iniziale
        msg_start = f"Avvio fallback scan (low-confidence recovery) su video={len(pending_videos)}"
        log.info(msg_start) # Logga su file
        #tqdm.write(f"[{time.strftime('%H:%M:%S')}] - ℹ️ {msg_start}") # Stampa pulito a schermo
        # Invece di tqdm.write manuale, puoi fare così:
        
        # 2. Progress Bar per la scansione YOLO a bassa confidenza
        pbar = tqdm(
            pending_videos,
            desc="Fallback YOLO",
            unit="video",
            ncols=100,
            bar_format="{desc}: {rate_fmt} [{bar}] {percentage:3.0f}% {n_fmt}/{total_fmt}"
        )
        #pbar.write(f"[{time.strftime('%H:%M:%S')}] - ℹ️ {msg_start}")
        for cam_id, record in pbar:
            v_path = record["video_path"]
            v_name = v_path.name
            images = record.get("nvr_images", [])
            
            for img_path in images:
                detection = self.yolo_engine.low_conf_image_scan(
                    image_path=img_path,
                    video_path=v_path,
                    cam_id=cam_id
                )
                
                if detection:
                    if v_name not in video_to_suspects:
                        video_to_suspects[v_name] = []
                    video_to_suspects[v_name].append(detection)
        
        pbar.close()


        # 3. Mandiamo alla Vision per la conferma finale
        if video_to_suspects:
            self._confirm_fallback_with_vision(video_to_suspects)


        # --- SALVATAGGIO STATS ---
        if self.is_test:
            # Qui contiamo quanti video abbiamo provato a recuperare
            self.stats["vision_fallback"]["count"] = len(pending_videos)
            # Il tempo include sia la scansione YOLO low-conf che il tempo speso in Vision
            self.stats["vision_fallback"]["time"] = time.time() - t_start
   
    def _confirm_fallback_with_vision(self, video_to_suspects):
        priority_order = ["person", "animal", "dog", "cat", "car", "motorcycle", "bus", "truck", "vehicle"]
        
        log.info(f"Processo Vision fallback recovery su video={len(video_to_suspects)}")

        msg = f"Processo Vision fallback recovery su video={len(video_to_suspects)}"
        
        # Se la barra non è ancora nata, tqdm.write si comporta come un print pulito
      
        tqdm.write(f"[{time.strftime('%H:%M:%S')}] - ℹ️ {msg}")
        
        pbar = tqdm(
            video_to_suspects.items(),
            desc="Vision Recovery",
            unit="it",
            bar_format="{desc}: {rate_fmt} [{bar}] {percentage:3.0f}% {n_fmt}/{total_fmt} {elapsed}"
        )

        for v_name, suspects in pbar:
            best_report = None
            current_priority_idx = len(priority_order)

            for suspect in suspects:
                # Assicurati che refine_fallback restituisca il dizionario col thinking
                report = self.vision_engine.refine_fallback(suspect)
                
                if not report:
                    continue

                # ESTRAZIONE SICURA DELLA CATEGORIA
                # Se report["category"] fosse accidentalmente un dict, lo castiamo a stringa
                raw_cat = report.get("category", "nothing")
                cat = str(raw_cat).lower() 
                
                # --- LOGGING THINKING (Solo se in test) ---
                if self.is_test and report.get("thinking"):
                    tqdm.write(f"\n{Fore.CYAN}{'='*40}{Style.RESET_ALL}")
                    tqdm.write(f"🔍 {Fore.MAGENTA}FALLBACK THINKING for {v_name}:{Style.RESET_ALL}")
                    tqdm.write(f"{Fore.LIGHTBLACK_EX}{report['thinking']}{Style.RESET_ALL}")
                    tqdm.write(f"🎯 {Fore.YELLOW}RECOVERY RESULT: {cat}{Style.RESET_ALL}")
                    tqdm.write(f"{Fore.CYAN}{'='*40}{Style.RESET_ALL}\n")

                if cat != "nothing":
                    # Calcolo priorità
                    priority_idx = priority_order.index(cat) if cat in priority_order else len(priority_order)
                    
                    if priority_idx == 0: # Person trovata
                        best_report = report
                        break
                    
                    if priority_idx < current_priority_idx:
                        current_priority_idx = priority_idx
                        best_report = report

            if best_report:
                self.final_reports.append(best_report)
                # Usiamo tqdm.write per non rompere la barra se il ciclo continua
                tqdm.write(f"✨ {Fore.GREEN}RECUPERATO:{Style.RESET_ALL} {v_name} -> {best_report.get('category')}")

        pbar.close()
    
    def _finalize_results(self, source_list, engine="yolo"):
        """
        Standardizza qualsiasi sorgente (YOLO results o Vision reports) 
        per il consumo da parte del Sorter.
        """
        standardized = []
        hierarchy = self.settings.get("classification_settings", {}).get("priority_hierarchy", ["person", "animal", "vehicle"])

        for item in source_list:
            v_path = Path(item["video_path"])
            cam_id = str(item["camera_id"])
            cam_name = self.cameras_config.get(cam_id, {}).get("name", f"Camera_{cam_id}")

            # Se l'item viene da Vision/Fallback, ha già la categoria decisa
            if engine == "vision_complex" or "category" in item:
                winner_cat = item["category"]
                best_frame = item.get("frame_priority") or item.get("best_frame_path")
                conf = item.get("confidence", 1.0)
                orig_engine = item.get("resolved_by", engine) # fallback_nvr o vision
            else:
                # Se viene da YOLO, dobbiamo calcolare il vincitore
                found = item.get("categories_found", {})
                winner_cat = "others"
                for cat in hierarchy:
                    if cat in found:
                        winner_cat = cat
                        break
                
                # Cerchiamo il frame migliore nella lista 'frames' di YOLO
                frames = item.get("frames", [])
                cat_frames = [f for f in frames if f["category"] == winner_cat]
                best_f_obj = max(cat_frames, key=lambda x: x["confidence"]) if cat_frames else None
                best_frame = best_f_obj["frame_path"] if best_f_obj else None
                conf = best_f_obj["confidence"] if best_f_obj else 0
                orig_engine = "yolo"

            standardized.append({
                "camera_id": cam_id,
                "camera_name": cam_name,
                "video_name": v_path.name,
                "video_path": str(v_path),
                "category": winner_cat,
                "confidence": conf,
                "best_frame_path": str(best_frame) if best_frame else None,
                "engine": orig_engine
            })
        return standardized
    
    def _clip_blip_scan_refine(self):
        self.clip_blip_results = []
        self.clip_blip_video_dict = {}
        t_start = time.time()  # <--- SPOSTATO QUI: deve essere all'inizio per tutti
        if not self.results:
            log.warning("No YOLO result to process.")
            return
        
        clip_blip_res_path = Path(self.output_dir) / CLIPBLIP_CACHE

        if clip_blip_res_path.exists():
            log.info("Clip/Blip Cache found. Resume...")
            self.clip_blip_video_dict = load_json(clip_blip_res_path)
        else:
            self.clip_blip_video_dict = {}

        # --- LOGICA LAZY LOADING ---
        # Controlliamo se c'è almeno un video che richiede l'analisi (non in cache e non scartato)
        videos_needing_analysis = [
            v for v in self.results 
            if v.get("video_path") not in self.clip_blip_video_dict 
            and "others" not in v.get("categories_found", [])
        ]

        if not videos_needing_analysis:
            log.debug("All videos are in cache. No need loading Blip engine.")
            # Se non ci sono video nuovi, usciamo o processiamo solo la cache
            #t_start = time.time()
            log.info(f"Start refine CLIP-BLIP on num_vid={len(self.results)}")
        else:
        # Carichiamo l'engine SOLO ORA che sappiamo di averne bisogno
            log.info(f"Loading CLIP-BLIP Engine to process {len(videos_needing_analysis)} videos.")
            
            self.clip_blip_engine = ClipBlipEngine(
                settings=self.settings,
                cameras_config=self.cameras_config,
                mode=self.mode,
                device=self.settings.get("yolo_settings", {}).get("device", "cpu")
            )
        pbar = tqdm(
            self.results,
            desc="CLIP-BLIP",
            unit="vid",
            ncols=100,
            bar_format="{desc}: {percentage:3.0f}% |{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        for video_data in self.results:
            cam_id_str = str(video_data["camera_id"])
            cam_info = self.cameras_config.get(cam_id_str, {})
            cam_name = cam_info.get("name", f"Camera_{cam_id_str}")
            video_path_key = video_data.get("video_path")

            # Inizializziamo raw_category a None per ogni ciclo
            raw_category = None

            # --- CASO A: Cache ---
            if video_path_key in self.clip_blip_video_dict:
                video_info = self.clip_blip_video_dict[video_path_key]
                raw_category = video_info.get("video_category")

            # --- CASO B: Scartati da YOLO ---
            elif "others" in video_data.get("categories_found", []):
                other_entry = {
                    "camera_id": cam_id_str,
                    "camera_name": cam_name, 
                    "video_name": Path(video_data["video_path"]).name,
                    "video_path": video_data["video_path"],
                    "category": "others",
                    "confidence": 0,
                    "best_frame_path": None,
                    "engine": "yolo discard",
                    "thinking": "Discarded by YOLO (No objects found)"
                }
                self.clip_blip_results.append(other_entry)
                pbar.update(1)
                continue 

            # --- CASO C: Analisi Reale ---
            else:
                video_dict = self.clip_blip_engine.scan_single_video(video_data)
                self.clip_blip_video_dict.update(video_dict)
                video_info = video_dict.get(video_path_key, {})
                raw_category = video_info.get("video_category")

            # --- CREAZIONE REPORT (Indentata correttamente sotto il for) ---
            if raw_category:
                final_category = raw_category.lower()
                if final_category != "empty":
                    best_frame = next(
                        (f for f in video_data["frames"] 
                        if f["category"].upper() == final_category.upper() and f.get("label") == final_category.upper()), 
                        video_data["frames"][0]
                    )

                    refined_entry = {
                        "camera_id": cam_id_str,
                        "camera_name": cam_name, 
                        "video_name": Path(video_data["video_path"]).name,
                        "video_path": video_data["video_path"],
                        "category": final_category,
                        "confidence": 1,
                        "best_frame_path": best_frame.get("frame_path"),
                        "engine": "clip_blip",
                        "thinking": f"Validated by clip_blip (Confirmed {final_category})"
                    }
                    self.clip_blip_results.append(refined_entry)
                
            # Update della barra dopo aver finito il processo (sia cache che AI)
            pbar.update(1)
            if self.is_real_time:
                save_json(self.clip_blip_video_dict, clip_blip_res_path)
                

        pbar.close()
        elapsed = time.time() - t_start
        
        # Salvataggio della cache tecnica (Sempre, non solo in is_test, così serve al Resume)
        save_json(self.clip_blip_video_dict, clip_blip_res_path)

        if self.is_test:
            self.stats["blip_analysis"] = {
                "count": len(self.results), 
                "confirmed": len(self.clip_blip_results),
                "time": elapsed
            }

        log.info(f"Refine complete in {elapsed:.2f}s. Valid_vids={len(self.clip_blip_results)}/{len(self.results)}")
  
    
    def _get_arbitration_queue(self):
        """
        Analizza i risultati di YOLO e BLIP per isolare i casi dubbi.
        Non servono parametri: legge self.results e self.clip_blip_results.
        """
        vision_queue = []
        # 1. Carichiamo la cache Vision per sapere chi ha già un verdetto
        vision_cache_file = Path(self.output_dir) / VISION_CACHE 

        processed_vision = {}
        if vision_cache_file.exists():
            cache_data = load_json(vision_cache_file)
            # Creiamo una mappa {path: report_completo}
            processed_vision = {str(r['video_path']): r for r in cache_data}
        # Creiamo la mappa dei risultati BLIP: {video_path: categoria}
        # Usiamo .lower() per evitare problemi di case-sensitivity
        blip_map = {str(r['video_path']): r['category'].lower() for r in self.clip_blip_results}

        for video_data in self.results:
            video_path = str(video_data.get("video_path"))
            # --- NOVITÀ: CONTROLLO CACHE PRIORITARIO ---
            if video_path in processed_vision:
                # Se è già in cache, aggiorniamo il verdetto in clip_blip_results 
                # così il report finale sarà corretto senza chiamare Qwen
                cache_entry = processed_vision[video_path]
                for i, r in enumerate(self.clip_blip_results):
                    if str(r["video_path"]) == video_path:
                        self.clip_blip_results[i] = cache_entry
                        break
                # SALTIAMO l'aggiunta alla coda: non serve ri-processarlo!
                continue
            
            yolo_cats = video_data.get("categories_found", [])
            
            # Recuperiamo il verdetto di BLIP (default a 'others' se non trovato)
            blip_cat = blip_map.get(video_path, "others")

            # --- LOGICA DI FILTRAGGIO ---

            # 1. Se YOLO ha visto una PERSONA, per noi è legge. Saltiamo Qwen.
            if "person" in yolo_cats:
                continue

            # 2. Se non ci sono persone, ma YOLO sospetta ANIMALI o VEICOLI
            has_animal = "animal" in yolo_cats
            has_vehicle = "vehicle" in yolo_cats

            if has_animal or has_vehicle:
                # CASO A: YOLO vede qualcosa, ma BLIP dice 'others' (il tuo filtro drastico)
                if blip_cat == "others":
                    vision_queue.append(video_data)
                
                # CASO B: Conflitto di specie (YOLO dice Animale, BLIP dice Persona)
                elif has_animal and blip_cat == "person":
                    vision_queue.append(video_data)
                
                # CASO C: BLIP conferma Animale. 
                # Visto che BLIP ha recall 9%, se ne vede uno vogliamo che Qwen confermi 
                # per evitare i falsi positivi (legna/alberi).
                elif blip_cat == "animal":
                    vision_queue.append(video_data)

        return vision_queue


    
    
    def _print_final_summary(self, total_time):
        # Conteggio delle categorie
        stats = {}
        for res in self.final_reports:
            cat = res.get('category', 'unknown')
            stats[cat] = stats.get(cat, 0) + 1
        
        # Calcolo dei totali
        total_videos = len(self.final_reports)
        
        # Separatore
        log.info("-" * 50)
        log.info("           Results")
        log.info("-" * 50)
        
        # Statistiche principali (usiamo il trigger '=' per i colori)
        log.info(f"⏱️  Total_time={total_time:.2f}s")
        log.info(f"🎥 Processed num_video={total_videos}")
        
        log.info("Category")
        
        for cat, count in stats.items():
            # Usiamo il trigger '='. Il logger colorerà la categoria in Cyan e il numero in Giallo
            log.info(f"  - category={cat.capitalize():<12} | count={count}")
        
        log.info("-" * 50)


    def check_cameras_clean(self):
        """
        Confronta l'immagine di riferimento in /checks con una delle immagini NVR 
        già presenti nel full_index (fascia 00:00 - 05:00).
        """
        log.info("🧼 Avvio 'Lens Health Check' (confronto immagini NVR)...")
        results = {}

        for cam_id, records in self.full_index.items():
            # 1. Recuperiamo il riferimento 'pulito' (es: checks/00.jpg)
            reference_img = self._get_reference_path(cam_id) # Utility che cerca cam_id.jpg/png
            if not reference_img:
                log.debug(f"Cam_id={cam_id}] No image found in folder=/checks for this camera.")
                continue

            # 2. Cerchiamo un'immagine NVR scattata di notte
            night_sample = None
            for rec in records:
                # Usiamo la tua funzione di parsing sul video_path del record
                _, timestamp = parse_filename(
                    rec["video_path"], 
                    self.settings["storage_settings"]["filename_template"],
                    self.settings["storage_settings"]["timestamp_format"]
                )
                
                if timestamp and 0 <= timestamp.hour <= 5:
                    # Se il record ha delle immagini NVR associate, ne prendiamo una
                    if rec["nvr_images"]:
                        night_sample = rec["nvr_images"][0] # Ne basta una
                        break
            
            # 3. Se abbiamo entrambi i file, interroghiamo il Vision Engine
            if night_sample:
                log.info(f"Cam_id={cam_id}] comparison with image={Path(night_sample).name}")
                
                # Mandiamo la lista [img1, img2] come richiesto
                status = self.vision_engine.analyze_cleanliness(
                    [str(reference_img), str(night_sample)], 
                    cam_id
                )
                results[cam_id] = status
            else:
                log.warning(f"Cam_id={cam_id}] No night nvr image found in folder={self.input_dir}")
                results[cam_id] = "unknown"

        return results

    def _find_night_video(self, records):
        """
        Cerca un video notturno usando il template e il formato data
        specificati dall'utente nelle impostazioni.
        """
        # Recuperiamo i parametri dai settings caricati
        template = self.settings["storage_settings"]["filename_template"]
        ts_format = self.settings["storage_settings"]["timestamp_format"]

        for record in records:
            video_path = Path(record["video_path"])
            
            # Chiamiamo la tua funzione di parsing
            cam_id, timestamp = parse_filename(video_path, template, ts_format)
            
            if timestamp:
                # Controllo finestra temporale: 00:00 - 05:59
                if 0 <= timestamp.hour <= 5:
                    return video_path
                    
        return None

    def _get_reference_path(self, cam_id):
        """
        Cerca l'immagine di riferimento (es. 02.jpg) nella cartella 'checks'.
        """
        # Usiamo la costante CHECKS_DIR che punta alla cartella /checks
        for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG"]:
            potential_ref = Path("checks") / f"{cam_id}{ext}"
            if potential_ref.exists():
                return potential_ref
        
        return None
    

    def _fallback_with_vision(self, vision_queue):
        if not vision_queue:
            log.debug("No results to process.")
            return
        
        vision_cache_file = Path(self.output_dir) / VISION_CACHE 

    
        # 1. CARICAMENTO CUMULATIVO (Non resettare a [])
        if vision_cache_file.exists():
            # Carichiamo i vecchi risultati per non perderli
            self.vision_results = load_json(vision_cache_file)
        else:
            self.vision_results = []

        # # Creiamo un set dei path già presenti per evitare duplicati nel file
        existing_paths = {str(r["video_path"]) for r in self.vision_results}
       

        msg_start = f"Start vision arbitration on num_vid={len(vision_queue)}"
        log.info(msg_start)

        # 2. Barra di progressione per i casi critici
        pbar = tqdm(
            vision_queue, 
            desc="⚖️  Vision Arbitration", 
            unit="it", 
            ncols=100,
            bar_format="{desc}: {rate_fmt} [{bar}] {percentage:3.0f}% {n_fmt}/{total_fmt} {elapsed}"
        )
        pbar.write(f"[{time.strftime('%H:%M:%S')}] - ℹ️ {msg_start}")
       
            
            # Controlliamo se i dati del video sono corretti (soprattutto i path)
            # pbar.write(f"DEBUG 📂: Path inviato: {video_path}")
        # 2. Creiamo una mappa per trovare velocemente i report in clip_blip_results
        # Assumiamo che clip_blip_results sia già popolata dallo scan precedente
        for video_data in pbar:
            video_path = video_data.get("video_path")
             # --- PRINT TATTICO 1: Ingresso ---
            # pbar.write(f"DEBUG 🔍: Invio a Vision Engine -> {video_path}")
            # pbar.write(f"DEBUG DATA 📦: {json.dumps(video_data, indent=2, default=str)}") # Usiamo json.dumps per vederlo ordinato
            # 3. Chiamata a Qwen solo per questo video
            report_vision = self.vision_engine.refine_single_video(video_data)
            
            if report_vision:
                # 4. SOSTITUZIONE CHIRURGICA
                # Cerchiamo il vecchio report di BLIP e lo rimpiazziamo con quello di Vision
                # pbar.write(f"DEBUG ✅: Risposta ricevuta per {video_path}")
                # pbar.write(f"DEBUG 🧠: Verdetto: {report_vision.get('category')} (Conf: {report_vision.get('confidence')})")
                for i, r in enumerate(self.clip_blip_results):
                    if r["video_path"] == video_path:
                        self.clip_blip_results[i] = report_vision
                        break

                # 3. AGGIUNTA ALLA LISTA CUMULATIVA
                if video_path not in existing_paths:
                    self.vision_results.append(report_vision)
                    existing_paths.add(video_path)
                
                # 4. SALVATAGGIO INCREMENTALE (Tutta la lista, non solo i nuovi)
                save_json(self.vision_results, vision_cache_file)
                
                # 5. Salvataggio incrementale della cache vision (opzionale ma consigliato)
                #self.vision_results.append(report_vision)
                # save_json(self.vision_results, Path(self.output_dir) / "vision_scan_res.json")
            # Logging del ragionamento se sei in modalità test
                if self.is_test and report_vision.get("thinking"):
                    pbar.write(f"\n🧠 Arbitro: {report_vision['video_name']} -> {report_vision['category']}")
        # 4. SALVATAGGIO INCREMENTALE (Tutta la lista, non solo i nuovi)
        save_json(self.vision_results, vision_cache_file)