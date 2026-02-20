from datetime import timedelta
import json
from pathlib import Path
import sys
import time

import torch
from smart_surveillance_sorter.constants import CAMERAS_JSON, CHECKS_DIR, PROJECT_ROOT, SETTINGS_JSON
import logging
from pathlib import Path
from smart_surveillance_sorter.file_sorter import FileSorter
from smart_surveillance_sorter.ram_monitor import ResourceMonitorAMD
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
    def __init__(self, mode, device=None, is_refine=False, is_fallback=False, is_test = False,engine="blip",is_check_clean=False):
       # 1. Inizializzazione fuori dal loop
        self.monitor = ResourceMonitorAMD()
        
        # 1. Parametri operativi
        self.mode = mode
        #self.device = device
        self.is_refine = is_refine
        self.is_fallback = is_fallback
        self.is_test = is_test
        self.engine = engine
        self.is_check_clean = is_check_clean
        # 2. Logger
        #log = logging.getLogger(self.__class__.__name__)



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

        log.info(f"🛠️ [Scanner] Inizializzato su device={self.device}")
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
            #logger=log
        )
        
        log.info(
            f"🚀 {self.__class__.__name__} inizializzato | "
            f"Mode={mode} | Refine={is_refine} | Device={self.device}"
        )
    
   
    def scan_folder(self, input_dir,output_dir):
        """
        Esegue il flusso completo: Scansione -> Analisi (YOLO/Vision) -> Finalizzazione -> Sorting.
        """
        # --- TIMER INIZIO ---
        t_start = time.time()
        self.monitor.log_stats("AVVIO SISTEMA")
        # 1. Normalizzazione Input
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # 2. Setup Directory di Lavoro
        self.frames_dir = self.output_dir / "extracted_frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"📂 Estrazione Frames in corso...  Directory={self.frames_dir}")

        # 3. Indicizzazione e Associazione
        raw_index = self._build_index(self.input_dir)
        self.full_index = self._associate_files(raw_index)

        # if self.engine == "vision" and self.is_check_clean:
        #     self.lens_status = self.check_cameras_clean()
            
        #     # Percorso del file JSON di report
        #     health_report_path = Path(self.output_dir) / "lens_health.json"
            
        #     try:
        #         with open(health_report_path, "w", encoding="utf-8") as f:
        #             json.dump(self.lens_status, f, indent=4)
        #         log.info(f"✅ Report salute lenti salvato in: {health_report_path}")
        #     except Exception as e:
        #         log.error(f"❌ Impossibile salvare il report lenti: {e}")


        # --- RECUPERO PEZZO CANCELLATO ---
        yolo_cache_file = Path(self.output_dir) / "yolo_scan_res.json"
        # 2. Check esistenza e caricamento/scansione
        if yolo_cache_file.exists():
            log.warning("♻️  Cache YOLO trovata! Caricamento risultati in corso...")
            with open(yolo_cache_file, 'r', encoding='utf-8') as f:
                self.results = json.load(f)
        else:
            log.info("🔍 Cache non trovata. Avvio scansione completa...")
            self._scan_images(self.full_index)
            self.monitor.log_stats(f"INIZIO SCANSIONE VIDEO YOLO")
            self._scan_videos(self.full_index)
            # Salviamo subito i risultati per i futuri test
            save_json(self.results, yolo_cache_file)
        
        # 4. Fase 1 & 2: Analisi YOLO (Popola self.results)
        # self._scan_images(self.full_index)
        # self._scan_videos(self.full_index)

        # 5. Fase 3: Raffinamento o Finalizzazione
        # Lo scopo qui è popolare 'final_data', l'unico oggetto che il Sorter leggerà.
        final_data = []
        self.final_reports = []
        self.monitor.log_stats("POST-YOLO / PRE-REFINE")
        if self.is_refine and self.engine == "blip":
            #... logica BLIPEngine ...
            from .clip_blip_engine import ClipBlipEngine # Import locale per velocità
            self.clip_blip_engine = ClipBlipEngine(
                settings=self.settings,
                cameras_config= self.cameras_config,
                mode = self.mode,
                device=self.settings.get("yolo_settings", {}).get("device", "cpu")
                #output_dir=self.output_dir
            )
            self.monitor.log_stats(f"INIZIO SCANSIONE VIDEO CLIPBLIP")
            self._clip_blip_scan_refine() #<---- non c'è bisogno di passargli niente e riempie results_clip
            log.info("Elaborazione verdetto finale YOLO + blip(no refine)...")
            self.final_reports = self.clip_blip_results
            final_data = self._finalize_results(self.final_reports, engine="blip")
            self.monitor.log_stats("FINE SESSIONE")
        elif self.is_refine and self.engine == "vision":
            
            # ... logica Vision AI ...
            log.info("✨ Avvio raffinamento con Vision AI...")
            self._ensure_vision_initialized() 

            if self.is_check_clean:
                self.lens_status = self.check_cameras_clean()
            
                # Percorso del file JSON di report
                health_report_path = Path(self.output_dir) / "lens_health.json"
                
                try:
                    with open(health_report_path, "w", encoding="utf-8") as f:
                        json.dump(self.lens_status, f, indent=4)
                    log.info(f"✅ Report salute lenti salvato in folder={health_report_path}")
                except Exception as e:
                    log.error(f"❌ Impossibile salvare il report lenti: errore={e}")
            # self._refine_with_vision popola internamente self.final_reports
            
            self._refine_with_vision() 
            final_data = self._finalize_results(self.final_reports, engine="vision")
            if self.is_fallback:
                # self._fallback_scan aggiunge altri record a self.final_reports
                self._fallback_scan() 
                final_data = self._finalize_results(self.final_reports, engine="vision")
            # self._ensure_vision_initialized()
            # self._refine_with_vision()
            # if self.is_fallback:
            #     self._fallback_scan()
            #     self.final_reports = self.final_reports
        
        
        # if not self.is_refine:
        #     # --- CASO SOLO YOLO + clip-
        #     from .blip_engine import BLIPEngine # Import locale per velocità
        #     self.blip_engine = BLIPEngine(
        #         settings=self.settings,
        #         #logger=log,
        #         device=self.settings.get("yolo_settings", {}).get("device", "cpu"),
        #         output_dir=self.output_dir
        #     )
        #     self._blip_scan_refine() #<---- non c'è bisogno di passargli niente e riempie results_clip
        #     log.info("Elaborazione verdetto finale YOLO + blip(no refine)...")
        #     self.final_reports = self.blip_results
        #     final_data = self._finalize_results(self.final_reports, engine="blip")
        # else:
        #     # --- CASO VISION AI (+ Fallback) ---
        #     log.info("✨ Avvio raffinamento con Vision AI...")
        #     self._ensure_vision_initialized() 
            
        #     # self._refine_with_vision popola internamente self.final_reports
        #     self._refine_with_vision() 
            
        #     if self.is_fallback:
        #         # self._fallback_scan aggiunge altri record a self.final_reports
        #         self._fallback_scan() 
        #         final_data = self._finalize_results(self.final_reports, engine="vision")
            # Trasformiamo i report accumulati nel formato standard final_data
            # Usiamo 'vision' come engine generico, i singoli record avranno 
            # poi il dettaglio (es. fallback_nvr) se necessario.
            #final_data = self._finalize_results(self.final_reports, engine="vision")
            

        # 6. Fase 4: Sorter (Il Gran Finale)
        if final_data:
            # Salvataggio unico del verdetto
            final_report_path = self.output_dir / "classification_results.json"
            save_json(final_data, final_report_path)
            log.info(f"📝 Report finale salvato in file={final_report_path}")
            
            # Inizializzazione e avvio Sorter
            self.file_sorter = FileSorter(
                self.settings, 
                self.input_dir, 
                self.output_dir, 
                self.is_test
            )
            
            # Il Sorter riceve il verdetto, i dati grezzi (per i frame) e l'indice (per NVR/Others)
            self.file_sorter.sort_all(
                final_results=final_data, 
                raw_results=self.results, 
                full_index=self.full_index
            )

            self.file_sorter.cleanup()

            total_time = time.time()-t_start
            if(self.is_test):
                save_test_metrics(output_dir, self.final_reports, total_time, self.stats, self.mode)
                
            self.monitor.log_stats("FINE SESSIONE")
        # 2. Pulisci memoria Python/Torch
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass
        
    def _build_index(self, input_dir):
      
        log.info(f"Avvio indicizzazione file in folder={input_dir}...")
        index = {}
        
        # Recuperiamo le impostazioni dal config
        # Se non esistono, usiamo dei default (Reolink style)
        storage_cfg = self.settings.get("storage_settings", {})
        template = storage_cfg.get("filename_template", "{nvr_name}_{camera_id}_{timestamp}")
        ts_format = storage_cfg.get("timestamp_format", "%Y%m%d%H%M%S")

        extensions = {".mp4", ".jpg", ".jpeg"}
        file_list = [f for f in input_dir.iterdir() if f.suffix.lower() in extensions]
        
      
        for f in file_list:
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
            file_type = "video" if f.suffix.lower() == ".mp4" else "image"
            
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
            log.info(f"Camera={cam_id}: video={vids} , img={imgs} immagini indicizzate.")
            
            if imgs < vids:
                log.warning(f"Camera={cam_id} ha meno immagini NVR rispetto ai video!")

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
        log.info("Avvio associazione Video-Immagini con logica di confine...")
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
            log.info("Nessuna immagine NVR da scansionare.")
            return

        # 1. INFO Iniziale
        msg_start = f"Avvio yolo scan su num_img={len(tasks)}"
        log.info(msg_start) # Logga su file
        #tqdm.write(f"[{time.strftime('%H:%M:%S')}] - ℹ️ {msg_start}") # Stampa pulito a schermo
        
        # Avvio Progress Bar
        pbar = tqdm(
            tasks,
            desc="Progress",
            unit="it",
            ncols=50,
            bar_format="{desc} {rate_fmt} [{bar}] {percentage:3.0f}% {n_fmt}/{total_fmt} {elapsed}<{remaining}"
        )
        pbar.write(f"[{time.strftime('%H:%M:%S')}] - ℹ️ {msg_start}")
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
            f"Processo yolo scan videos model={model_name},device={self.device}")

        # 2. Prepariamo i compiti (tasks) per la barra
        tasks = []
        for cam_id, items in associations.items():
            for item in items:
                video_path = item["video_path"] # Path object
                video_str = str(video_path)

                # Includiamo solo se non risolto e se il file esiste
                if video_str not in self.resolved_videos and video_path.exists():
                    tasks.append((cam_id, item, video_path, video_str))
                # # Skip se già risolto dalle immagini NVR
                # if video_str in self.resolved_videos:
                #     continue

                # if not video_path.exists():
                #     continue
        if not tasks:
            log.info("Nessun video da scansionare (tutti risolti o mancanti).")
            return    

        # 1. INFO Iniziale
        msg_start = f"Avvio yolo scan su video={len(tasks)}{Style.RESET_ALL}"
        log.info(msg_start) # Logga su file
        #tqdm.write(f"[{time.strftime('%H:%M:%S')}] - ℹ️ {msg_start}") # Stampa pulito a schermo
        # Invece di tqdm.write manuale, puoi fare così:
        
        # 3. Avvio Progress Bar
        # Progress 1,59s/it [barra] 100% num/tot
        pbar = tqdm(
            tasks,
            desc="Progress",
            unit="it",
            ncols=50,
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
        pbar.close()
        # --- SALVATAGGIO STATS ---
        if self.is_test:
            self.stats["yolo_videos"]["count"] = len(tasks)
            self.stats["yolo_videos"]["time"] = time.time() - t_start
    
        # 4. Conteggio finale (opzionale, ma utile)
        log.info(f"Video processati con successo.")
    
    def _refine_with_vision(self):
        if not self.results:
            log.warning("Nessun risultato YOLO da raffinare.")
            return
        # --- TIMER INIZIO ---
        t_start = time.time()
        # 1. INFO iniziale con modello (come richiesto)
        log.info(f"Processo Vision refine")

        # 2. Prepariamo i report finali
        self.final_reports = []
        # 1. INFO Iniziale
        msg_start = f"Avvio refine su img={len(self.results)}"
        log.info(msg_start) # Logga su file
        #tqdm.write(f"[{time.strftime('%H:%M:%S')}] - ℹ️ {msg_start}") # Stampa pulito a schermo
        # Invece di tqdm.write manuale, puoi fare così:
        
        # 3. Avvio Progress Bar
        # Usiamo self.results come iteratore per la barra
        pbar = tqdm(
            self.results,
            desc="Progress",
            unit="it",
            ncols=50,
            bar_format="{desc}: {rate_fmt} [{bar}] {percentage:3.0f}% {n_fmt}/{total_fmt} {elapsed}<{remaining}"
        )
        pbar.write(f"[{time.strftime('%H:%M:%S')}] - ℹ️ {msg_start}")
        for video_data in pbar:
            # Chiamata al motore Vision
            report = self.vision_engine.refine_single_video(video_data)
            self.final_reports.append(report)

            # --- LOGGING DEL THINKING SOLO IN TEST ---
            if self.is_test and report.get("thinking"):
                tqdm.write(f"\n{Fore.CYAN}{'='*40}{Style.RESET_ALL}")
                tqdm.write(f"🧠 {Fore.MAGENTA}REASONING for {report['video_name']}:{Style.RESET_ALL}")
                # Usiamo un colore spento (LIGHTBLACK) per il testo lungo così non affatica
                tqdm.write(f"{Fore.LIGHTBLACK_EX}{report['thinking']}{Style.RESET_ALL}")
                tqdm.write(f"🎯 {Fore.GREEN}FINAL VERDICT: {report['category']}{Style.RESET_ALL}")
                tqdm.write(f"{Fore.CYAN}{'='*40}{Style.RESET_ALL}\n")

        pbar.close()
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

        log.info("🧠 Preparazione motore Vision AI (Ollama)...")
        
        try:
            from smart_surveillance_sorter.scanners.vision_engine import VisionEngine
            
            self.vision_engine = VisionEngine(
                settings=self.settings,
                cameras_config=self.cameras_config,
                mode=self.mode
                #logger=log
            )

            log.info("✅ Modulo Vision AI pronto.")
            
        except ImportError as e:
            log.error(f"❌ Errore nell'import di VisionAnalyzer: error={e}")
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
            ncols=50,
            bar_format="{desc}: {rate_fmt} [{bar}] {percentage:3.0f}% {n_fmt}/{total_fmt}"
        )
        pbar.write(f"[{time.strftime('%H:%M:%S')}] - ℹ️ {msg_start}")
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
      
        if not self.results:
            log.warning("Nessun risultato YOLO da passare a BLIP.")
            return

        t_start = time.time()

        log.info(f"🚀 Inizio raffinamento CLIP-BLIP su num_video={len(self.results)}")

        # Inizializziamo la barra
        pbar = tqdm(
            self.results,
            desc="CLIP-BLIP",
            unit="vid",
            ncols=100, # Larghezza totale contenuta
            bar_format="{desc}: {percentage:3.0f}% |{bar:20}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        for video_data in self.results:
            # 1. Recupero informazioni telecamera per il formato finale
            cam_id_str = str(video_data["camera_id"])
            # Cerchiamo la camera nel config (che dovresti avere nello Scanner)
            cam_info = self.cameras_config.get(cam_id_str, {})
            cam_name = cam_info.get("name", f"Camera_{cam_id_str}")
      
            video_dict = self.clip_blip_engine.scan_single_video(video_data)
            # Recuperi il path del video (che è la chiave del dizionario restituito)
            video_path_key = video_data.get("video_path")

            # Accedi ai dati del video usando la chiave
            video_info = video_dict.get(video_path_key, {})

            # Ora puoi prendere la categoria in modo sicuro
            raw_category = video_info.get("video_category")
            #print(f"DEBUG: Raw category: {raw_category} | Type: {type(raw_category)}")
            final_category = raw_category.lower()
            if final_category != "empty":
                # 2. Troviamo il "best_frame" 
                best_frame = next(
                    (f for f in video_data["frames"] 
                    if f["category"].upper() == final_category.upper() and f.get("label") == final_category.upper()), 
                    video_data["frames"][0] # Fallback sul primo frame se è un override YOLO
                )

                # 3. Costruiamo il dizionario esattamente come lo vuole la funzione di spostamento
                refined_entry = {
                    "camera_id": str(video_data["camera_id"]),
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
                #log.info(f"✅ Video_validato={refined_entry['video_name']} -> categoria={final_category}")
                # Usiamo pbar.write per non rompere la barra grafica
                #pbar.write(f"  ✅ video={refined_entry['video_name']} -> cat={final_category}")
                self.clip_blip_video_dict.update(video_dict)
                # --- AGGIUNGI QUESTA RIGA QUI SOTTO ---
                pbar.update(1)
        # 3. Chiusura barra e statistiche
        pbar.close()
        # 3. Aggiorniamo le statistiche se in modalità test
        elapsed = time.time() - t_start
        if self.is_test:
            self.stats["blip_analysis"] = {
                "count": len(self.results), 
                "confirmed": len(self.clip_blip_results),
                "time": elapsed
            }
            clip_blip_res_path = Path(self.output_dir) / "clip_blip_res.json"
            with open(clip_blip_res_path, "w") as f:
                json.dump(self.clip_blip_video_dict, f, indent=4)

        log.info(f"🏁 Raffinamento CLIP-BLIP completato in time={elapsed:.2f}s. Video_validi={len(self.clip_blip_results)}/{len(self.results)}")
    
    
    # def _blip_scan_refine(self):
    #     self.blip_results = []

    #     if not self.results:
    #         log.warning("Nessun risultato YOLO da passare a BLIP.")
    #         return

    #     t_start = time.time()
        
    #     for video_data in self.results:
    #         # 1. Recupero informazioni telecamera per il formato finale
    #         cam_id_str = str(video_data["camera_id"])
    #         # Cerchiamo la camera nel config (che dovresti avere nello Scanner)
    #         cam_info = self.cameras_config.get(cam_id_str, {})
    #         cam_name = cam_info.get("name", f"Camera_{cam_id_str}")
    #         # 1. Chiamiamo l'analisi (ritorna "person", "animal", "vehicle" o "empty")
    #         final_category = self.blip_engine.analyze_video_results(video_data)
            
    #         if final_category != "empty":
    #             # 2. Troviamo il "best_frame" (il primo che ha confermato la categoria)
    #             # Se BLIP ha confermato più frame, prendiamo il primo che ha 'blip_confirmed' = True
    #             best_frame = next(
    #                 (f for f in video_data["frames"] 
    #                 if f["category"].upper() == final_category.upper() and f.get("blip_confirmed")), 
    #                 video_data["frames"][0] # Fallback sul primo frame se è un override YOLO
    #             )

    #             # 3. Costruiamo il dizionario esattamente come lo vuole la funzione di spostamento
    #             refined_entry = {
    #                 "camera_id": str(video_data["camera_id"]),
    #                 "camera_name": video_data.get("camera_name", "Unknown"), # Se lo hai nei dati originali
    #                 "video_name": Path(video_data["video_path"]).name,
    #                 "video_path": video_data["video_path"],
    #                 "category": final_category,
    #                 "confidence": best_frame.get("blip_confidence", best_frame["confidence"]),
    #                 "best_frame_path": best_frame.get("frame_path"),
    #                 "engine": "yolo_blip",
    #                 "thinking": f"Validated by BLIP (Confirmed {final_category})"
    #             }

    #             self.blip_results.append(refined_entry)
    #             log.info(f"✅ Video validato: {refined_entry['video_name']} -> {final_category}")
        
    #     # 3. Aggiorniamo le statistiche se in modalità test
    #     elapsed = time.time() - t_start
    #     if self.is_test:
    #         self.stats["blip_analysis"] = {
    #             "count": len(self.results), 
    #             "confirmed": len(self.blip_results),
    #             "time": elapsed
    #         }
        
    #     log.info(f"🏁 Raffinamento BLIP completato in {elapsed:.2f}s. Video validi: {len(self.blip_results)}/{len(self.results)}")
    
    
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
        log.info("           RIEPILOGO ANALISI FINALE")
        log.info("-" * 50)
        
        # Statistiche principali (usiamo il trigger '=' per i colori)
        log.info(f"⏱️  Tempo totale={total_time:.2f}s")
        log.info(f"🎥 Video processati={total_videos}")
        
        log.info("Suddivisione categorie:")
        
        for cat, count in stats.items():
            # Usiamo il trigger '='. Il logger colorerà la categoria in Cyan e il numero in Giallo
            log.info(f"  - categoria={cat.capitalize():<12} | conteggio={count}")
        
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
                log.debug(f"Cam_id={cam_id}] Nessun riferimento in /checks, salto il controllo.")
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
                log.info(f"Cam_id={cam_id}] Confronto riferimento con immagine_NVR={Path(night_sample).name}")
                
                # Mandiamo la lista [img1, img2] come richiesto
                status = self.vision_engine.analyze_cleanliness(
                    [str(reference_img), str(night_sample)], 
                    cam_id
                )
                results[cam_id] = status
            else:
                log.warning(f"Cam_id={cam_id}] Nessuna immagine NVR notturna trovata nell'indice.")
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