from datetime import timedelta
from pathlib import Path
import sys
import time
import torch
from smart_surveillance_sorter.constants import CAMERAS_JSON, CHECKS_DIR, CLIPBLIP_CACHE, CLIPBLIP_FALLBACK_CACHE, FINAL_REPORT, FRAME_DIR, LENS_HEALTH, SETTINGS_JSON, VISION_CACHE, YOLO_CACHE
import logging
from smart_surveillance_sorter.file_utils import associate_files, build_index, sortVideos, parse_filename
from smart_surveillance_sorter.logger import  get_pbar_prefix, log_resource_usage
from smart_surveillance_sorter.scanners.clip_blip_engine import ClipBlipEngine
from smart_surveillance_sorter.scanners.yolo_engine import YoloEngine
from smart_surveillance_sorter.utils import load_json, save_json, save_test_metrics, validate_ollama_setup
from tqdm import tqdm
from colorama import Fore, Style


log = logging.getLogger(__name__)

class Scanner():
    def __init__(self, mode, device=None, is_refine=False, is_fallback=False, is_test = False,engine="blip",is_check_clean=False,is_real_time=False,is_sort=True):     

        self.mode = mode
        self.is_refine = is_refine
        self.is_fallback = is_fallback
        self.is_test = is_test
        self.engine = engine
        self.is_check_clean = is_check_clean
        self.is_real_time = is_real_time
        self.is_sort = is_sort

        if self.is_test:
            self.stats = {
            "yolo_images": {"count": 0, "time": 0},
            "yolo_videos": {"count": 0, "time": 0},
            "vision_refine": {"count": 0, "time": 0},
            "vision_fallback": {"count": 0, "time": 0}
            }

        self.settings              =  load_json(SETTINGS_JSON)
        self.cameras_config        =  load_json(CAMERAS_JSON)

        if self.engine == "vision" or self.is_fallback:
            self.vision_cfg = self.settings.get("vision_settings", {})
            if not validate_ollama_setup(self.vision_cfg):
                sys.exit(1) 

        yolo_cfg = self.settings.get("yolo_settings", {})
        
        if device is not None:
            self.device = device
        else:
            self.device = yolo_cfg.get("device", "cpu")

        log_resource_usage(log, "START")
        
        # 3. Stato dell'analisi
        self.skipped_index = {}
        self.results = []               # Risultati pronti per il report/sorting
        self.resolved_videos = set()    # Per evitare di ri-analizzare video già risolti
        self.frames_dir = None          
        self.final_reports = []
        self.vision_engine = None

        # Istanza del motore 
        self.yolo_engine = YoloEngine(
            mode=self.mode,
            device=self.device,
            settings=self.settings,
            cameras_config=self.cameras_config
            
        )

    def scan_folder(self, input_dir,output_dir):
        """
        Esegue il flusso completo: Scansione -> Analisi (YOLO/Vision) -> Finalizzazione -> Sorting.
        """
        # --- TIMER INIZIO ---
        t_start = time.time()

        # 1. Normalizzazione Input
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        

        if not self._check_engine_integrity():
        # Solleviamo un errore bloccante
            raise RuntimeError("Engine changed between scans! Delete results json or use the correct engine to resume/continue.")
        # 2. Setup Directory di Lavoro
        self.frames_dir = self.output_dir / FRAME_DIR
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        log.debug(f"Frames directory={self.frames_dir}")

        # 3. Indicizzazione e Associazione
        raw_index = build_index(input_dir,self.settings)
        self.full_index = associate_files(raw_index,self.input_dir)
        
        #4. controlliamo se esistono risultati di yolo precedenti per il resume
        self._handle_yolo_cache()
        
        # 4. Scansione yolo (sempre)
        if self.new_video_count > 0:
            if not self.processed_set:
                log.info("NVR Images scan for person.")
                self._yolo_scan_images(self.full_index)
            
            log.info(f"YOLO scan on {self.new_video_count} videos")
            log_resource_usage(log, "YOLO")
            self._yolo_scan_videos(self.full_index)
        else:
            log.info("All video processed.")

        #ora bisogna dividere il flusso yolo -> blip oppure yolo -> vision
        # Lo scopo qui è popolare 'final_data', l'unico oggetto che il Sorter leggerà.
        self.final_data = []
        self.final_reports = []

        if self.is_refine and self.engine=="blip":
            #funzione corso blip 
            self._clip_blip_scan()
        if self.is_refine and self.engine=="vision":
            #funzione corso vision
            self._vision_scan()

        #abbiamo final data e final_reports passiamo al sorter
        # 6. Fase 4: Sorter 
        if self.final_data:
            # Salvataggio unico del verdetto
            final_report_path = self.output_dir / FINAL_REPORT
            save_json(self.final_data, final_report_path)
            log.info(f"Final results saved in file={final_report_path}")
            
            #controlliamo se dobbiamo organizzare i video
            if self.is_sort:
                # Reintegra i video skippati nel full_index per il sorter
                for cam_id, entries in self.skipped_index.items():
                    if cam_id in self.full_index:
                        self.full_index[cam_id].extend(entries)
                    else:
                        self.full_index[cam_id] = entries
                sortVideos(self.settings,self.input_dir,self.output_dir,
                           self.is_test,self.final_data,self.results,
                           self.full_index)

        total_time = time.time()-t_start
        log.info(f"Folder scan in {total_time}s")
        #se è test dobbiamo salvare le metriche in un json
        if(self.is_test):
            save_test_metrics(output_dir, self.final_reports, total_time, self.stats, self.mode,self.settings)

    def _clip_blip_scan(self):
        #yolo -> blip
        self._clip_blip_scan_refine()
        if self.is_fallback:
            #blip -> vision
            #caricamento dei casi dubbi da passare
            vision_queue = self._get_arbitration_queue()
            if vision_queue:
                    log.info("Fallback step with vision on blip results")
                    log_resource_usage(log, "VISION")
                    #inizializziamo vision
                    self._ensure_vision_initialized() 
                    log.info(f"{len(vision_queue)} to process.")
                    # Passiamo la coda filtrata
                    self._fallback_blip_vision(vision_queue)
        #in ogni caso  abbiamo il risultato
        log.info("Scan end. Start folder sorting.")
        self.final_reports = self.clip_blip_results
        self.final_data = self._finalize_results(self.final_reports, engine="blip")
        log_resource_usage(log, "END")
    
    def _vision_scan(self):
        #yolo -> vision
        log.info("Starting refine with vision")
        #controlliamo le lenti --check-clean    
        if self.is_check_clean:
            log_resource_usage(log, "VISION")
            self.lens_status = self.check_cameras_clean()
            # Percorso del file JSON di report
            health_report_path = Path(self.output_dir) / LENS_HEALTH
            save_json(self.lens_status, health_report_path)
            log.info(f"Cameras lens status report saved in folder={health_report_path}")
          
        # self._refine_with_vision popola internamente self.final_reports
        log_resource_usage(log, "VISION")
        #yolo -> vision (refine)
        self._vision_scan_refine()
        #controllo se --fallback anche su --vision
        if self.is_fallback:
            self._fallback_vision_vision()
        #in ogni caso abbiamo final reports
        self.final_data = self._finalize_results(self.final_reports, engine="vision")
        log_resource_usage(log, "END VISION")

    def _handle_yolo_cache(self):
        #recupero eventuale yolo cache per resume/skip
        yolo_cache_file = Path(self.output_dir) / YOLO_CACHE
        self.yolo_cache_file = yolo_cache_file

        #se la cache esiste carichiamo i dati in self results.
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
        
        self.processed_set = {str(r.get("video_path")) for r in self.results if isinstance(r, dict) and r.get("video_path")}

        # Salva i video già processati in un indice separato
        self.skipped_index = {
            cam_id: [
                v for v in v_list
                if str(v.get("video_path")) in self.processed_set
            ]
            for cam_id, v_list in self.full_index.items()
        }
        # Filtriamo mantenendo la struttura
        self.full_index = {
            cam_id: [
                v for v in v_list 
                if str(v.get("video_path")) not in self.processed_set
            ]
            for cam_id, v_list in self.full_index.items()
        }

        # Ricalcoliamo il totale reale dei video rimasti
        self.new_video_count = sum(len(v_list) for v_list in self.full_index.values())
        self.skipped = original_video_count - self.new_video_count

    def _handle_clip_blip_cache(self):
        self.clip_blip_res_path = Path(self.output_dir) / CLIPBLIP_CACHE

        if self.clip_blip_res_path.exists():
            log.info("Clip/Blip Cache found. Resume...")
            self.clip_blip_video_dict = load_json(self.clip_blip_res_path)
        else:
            self.clip_blip_video_dict = {}

        videos_needing_analysis = [
            v for v in self.results 
            if v.get("video_path") not in self.clip_blip_video_dict 
            and "others" not in v.get("categories_found", [])
        ]
        return videos_needing_analysis

    def _yolo_scan_images(self, associations):
        t_start = time.time()
     
        tasks = []
        for cam_id, items in associations.items():
            for item in items:
                video_path = str(item["video_path"])
                nvr_images = item.get("nvr_images", [])

                # Filtro di inclusione per la lista tasks
                if nvr_images and video_path not in self.resolved_videos:
                    tasks.append((cam_id, item, nvr_images))
              
        # Se non ci sono immagini NVR da analizzare, usciamo subito
        if not tasks:
            log.info("No nvr images to scan")
            return
        self.yolo_engine.ensure_model_loaded()
       
        msg_start = f"Yolo scan on num_img={len(tasks)}"
        log.info(msg_start) # Logga su file
       
        
        # Avvio Progress Bar
        pbar = tqdm(
            tasks,
            desc="Progress",
            unit="img",
            ncols=100,
            bar_format=f"{get_pbar_prefix('YOLO Scan')} {{rate_fmt:>10}} [{{bar}}] {{percentage:3.0f}}% {{n_fmt}}/{{total_fmt}} {{elapsed}}<{{remaining}}"
        )
        
        count_resolved = 0
        for cam_id, item, nvr_images in pbar:
            video_path = str(item["video_path"])
            
            for img_path in nvr_images:
                result = self.yolo_engine.scan_single_image(img_path, item["video_path"], self.frames_dir, cam_id)
                if result:
                    self.results.append(result)
                    self.resolved_videos.add(video_path)
                    count_resolved += 1
                    break 
      
        pbar.close()
        
        if self.is_test:
            self.stats["yolo_images"]["count"] = len(tasks)
            self.stats["yolo_images"]["time"] = time.time() - t_start

    def _yolo_scan_videos(self, associations):
        
       
        t_start = time.time()
       
        model_name = self.settings.get("yolo_settings").get("model_path")
        log.info(
            f"Process yolo scan videos model={model_name},device={self.device}")

        
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
        self.yolo_engine.ensure_model_loaded()
        
        msg_start = f"Yolo scan on num_vid={len(tasks)}{Style.RESET_ALL}"
        log.info(msg_start) 
    
        
        pbar = tqdm(
            tasks,
            desc="Progress",
            unit="vid",
            ncols=100,
            bar_format=f"{get_pbar_prefix('YOLO Scan')} {{rate_fmt:>10}} [{{bar}}] {{percentage:3.0f}}% {{n_fmt}}/{{total_fmt}} {{elapsed}}<{{remaining}}"
           
         )
        
        for cam_id, item, video_path, video_str in pbar:       
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
        
        if self.is_test:
            self.stats["yolo_videos"]["count"] = len(tasks)
            self.stats["yolo_videos"]["time"] = time.time() - t_start
    
        log.info(f"All videos processed.")

    def _clip_blip_scan_refine(self):
        self.clip_blip_results = []
        self.clip_blip_video_dict = {}
        
        if not self.results:
            log.warning("No YOLO result to process.")
            return
        t_start = time.time() 
        
        #eventuale resume
        videos_needing_analysis = self._handle_clip_blip_cache()
        
        if not videos_needing_analysis:
            log.debug("All videos are in cache. No need loading Blip engine.")
        else:
            # Lazy loading we need the engine
            log.info(f"Loading CLIP-BLIP Engine to process {len(videos_needing_analysis)} videos.")
            
            self.clip_blip_engine = ClipBlipEngine(
                settings=self.settings,
                cameras_config=self.cameras_config,
                mode=self.mode,
                device=self.device
            )
        pbar = tqdm(
            self.results,
            desc="CLIP-BLIP",
            unit="vid",
            ncols=100,
            bar_format=f"{get_pbar_prefix('Blip Scan')} {{rate_fmt:>10}} [{{bar}}] {{percentage:3.0f}}% {{n_fmt}}/{{total_fmt}} {{elapsed}}<{{remaining}}"
        )
        
        for video_data in self.results:
            cam_id_str = str(video_data["camera_id"])
            cam_info = self.cameras_config.get(cam_id_str, {})
            cam_name = cam_info.get("name", f"Camera_{cam_id_str}")
            video_path_key = video_data.get("video_path")

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

            # --- CREAZIONE REPORT 
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
                
           
            pbar.update(1)
            
            save_json(self.clip_blip_video_dict, self.clip_blip_res_path)
                

        pbar.close()
        
        
        save_json(self.clip_blip_video_dict, self.clip_blip_res_path)
        elapsed = time.time() - t_start
       
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

        # 1. Set dei video "attivi" nel ciclo corrente (full_index + skipped_index)
        active_videos = set()
        for cam_id, entries in self.full_index.items():
            for entry in entries:
                active_videos.add(str(entry["video_path"]))
        for cam_id, entries in self.skipped_index.items():
            for entry in entries:
                active_videos.add(str(entry["video_path"]))

        # 2. Cache Vision per resume
        vision_cache_file = Path(self.output_dir) / VISION_CACHE
        processed_vision = {}
        if vision_cache_file.exists():
            cache_data = load_json(vision_cache_file)
            processed_vision = {str(r['video_path']): r for r in cache_data}

        # 3. Mappa risultati BLIP
        blip_map = {str(r['video_path']): r['category'].lower() for r in self.clip_blip_results}

        for video_data in self.results:
            video_path = str(video_data.get("video_path"))

            # Skippa video non attivi (già ordinati in cicli precedenti)
            if video_path not in active_videos:
                continue

            # Cache Vision: già processato, aggiorna clip_blip_results e skippa
            if video_path in processed_vision:
                cache_entry = processed_vision[video_path]
                for i, r in enumerate(self.clip_blip_results):
                    if str(r["video_path"]) == video_path:
                        self.clip_blip_results[i] = cache_entry
                        break
                continue

            yolo_cats = video_data.get("categories_found", [])
            blip_cat  = blip_map.get(video_path, "others")

            # YOLO ha visto PERSON → legge, skip
            if "person" in yolo_cats:
                continue

            has_animal  = "animal"  in yolo_cats
            has_vehicle = "vehicle" in yolo_cats

            if has_animal or has_vehicle:
                # CASO A: YOLO vede qualcosa, BLIP dice others
                if blip_cat == "others":
                    vision_queue.append(video_data)
                # CASO B: Conflitto YOLO=animal, BLIP=person
                elif has_animal and blip_cat == "person":
                    vision_queue.append(video_data)
                # CASO C: BLIP conferma animal (recall 9%, vogliamo conferma Qwen)
                elif blip_cat == "animal":
                    vision_queue.append(video_data)

        return vision_queue


    def _fallback_blip_vision(self, vision_queue):
        if not vision_queue:
            log.debug("No results to process.")
            return
        
        vision_cache_file = Path(self.output_dir) / CLIPBLIP_FALLBACK_CACHE

        # 1. CARICAMENTO CUMULATIVO 
        if vision_cache_file.exists():
            # Carichiamo i vecchi risultati per non perderli
            self.vision_results = load_json(vision_cache_file)
        else:
            self.vision_results = []

        # # Creiamo un set dei path già presenti per evitare duplicati nel file
        existing_paths = {str(r["video_path"]) for r in self.vision_results}
       
        msg_start = f"Start vision arbitration on num_vid={len(vision_queue)}"
        log.info(msg_start)
        #inizializza vision   
        self._ensure_vision_initialized() 
        
        pbar = tqdm(
            vision_queue, 
            desc="⚖️  Vision Arbitration", 
            unit="vid", 
            ncols=100,
            bar_format=f"{get_pbar_prefix('Vision Scan')} {{rate_fmt:>10}} [{{bar}}] {{percentage:3.0f}}% {{n_fmt}}/{{total_fmt}} {{elapsed}}<{{remaining}}"
        )
       
        for video_data in pbar:
            video_path = video_data.get("video_path")
            
            report_vision = self.vision_engine.refine_single_video(video_data)
            
            if report_vision:
                # Cerchiamo il vecchio report di BLIP e lo rimpiazziamo con quello di Vision
                for i, r in enumerate(self.clip_blip_results):
                    if r["video_path"] == video_path:
                        self.clip_blip_results[i] = report_vision
                        break

                # 3. AGGIUNTA ALLA LISTA CUMULATIVA
                if video_path not in existing_paths:
                    self.vision_results.append(report_vision)
                    existing_paths.add(video_path)
                
                # 4. SALVATAGGIO INCREMENTALE 
                save_json(self.vision_results, vision_cache_file)
                
                # Logging del ragionamento 
                if self.is_test and report_vision.get("thinking"):
                    pbar.write(f"Thinking: {report_vision.get("thinking")}")
                    pbar.write(f"\n🧠 Arbitration: {report_vision['video_name']} -> {report_vision['category']}")

        # 4. SALVATAGGIO finale
        save_json(self.vision_results, vision_cache_file)
    
    def _vision_scan_refine(self):
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
        
     
        t_start = time.time()

        # 2. Prepariamo i report finali
        self.final_reports = []
      
        msg_start = f"Start refine on num_vid={len(self.results)}"
        log.info(msg_start) # Logga su file
        self._ensure_vision_initialized() 

        with tqdm(
            self.results,
            desc="Vision Refine",
            unit="vid", 
            ncols=100,
            bar_format=f"{get_pbar_prefix('Vision Scan')} {{rate_fmt:>10}} [{{bar}}] {{percentage:3.0f}}% {{n_fmt}}/{{total_fmt}} {{elapsed}}<{{remaining}}"
        ) as pbar:
  
          
           
            for video_data in pbar:
                video_path = str(video_data.get("video_path"))

                # --- 1. SALTO SE GIÀ IN CACHE ---
                if video_path in processed_vision_paths:
                    cache_entry = next((r for r in self.vision_results if str(r["video_path"]) == video_path), None)
                    if cache_entry:
                        self.vision_results.append(cache_entry)
                   
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
                    
                    save_json(self.vision_results, vision_cache_file)
                    
                    continue 

                # --- 3. CHIAMATA AL MOTORE VISION (QWEN) ---
                report = self.vision_engine.refine_single_video(video_data)
                
                if report:
                    self.vision_results.append(report)
                    # SALVATAGGIO INCREMENTALE 
                    save_json(self.vision_results, vision_cache_file)

                    # --- LOGGING TEST ---
                    if self.is_test and report.get("thinking"):
                        tqdm.write(f"\n{Fore.CYAN}{'='*40}{Style.RESET_ALL}")
                        tqdm.write(f"🧠 {Fore.MAGENTA}REASONING for {report['video_name']}:{Style.RESET_ALL}")
                        tqdm.write(f"{Fore.LIGHTBLACK_EX}{report['thinking']}{Style.RESET_ALL}")
                        tqdm.write(f"🎯 {Fore.GREEN}FINAL VERDICT: {report['category']}{Style.RESET_ALL}")
                        tqdm.write(f"{Fore.CYAN}{'='*40}{Style.RESET_ALL}\n")
                
              

        
        self.final_reports = self.vision_results
       
        if self.is_test:
            self.stats["vision_refine"]["count"] = len(self.results)
            self.stats["vision_refine"]["time"] = time.time() - t_start
    
    def _fallback_vision_vision(self):

     
        t_start = time.time()
        identified_video_names = {res['video_name'] for res in self.final_reports}
        
        
        video_to_suspects = {} 
        
        # Prepariamo la lista dei video da analizzare per la barra
        pending_videos = []
        for cam_id, records in self.full_index.items():
            for record in records:
                if record["video_path"].name not in identified_video_names:
                    pending_videos.append((cam_id, record))

        if not pending_videos:
            log.info("No video to check in fallback phase.")
            return
        # 1. INFO Iniziale
        msg_start = f"Start fallback scan (low-confidence recovery) su video={len(pending_videos)}"
        log.info(msg_start)
        
      
        pbar = tqdm(
            pending_videos,
            desc="Fallback YOLO",
            unit="img",
            ncols=100,
            bar_format=f"{get_pbar_prefix('YOLO Scan')} {{rate_fmt:>10}} [{{bar}}] {{percentage:3.0f}}% {{n_fmt}}/{{total_fmt}} {{elapsed}}<{{remaining}}"
        )
      
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
            self._confirm_fallback_vision_vision(video_to_suspects)

      
        if self.is_test:
            self.stats["vision_fallback"]["count"] = len(pending_videos)
            self.stats["vision_fallback"]["time"] = time.time() - t_start

    def _confirm_fallback_vision_vision(self, video_to_suspects):
        self._ensure_vision_initialized() 
        priority_order = ["person", "animal", "dog", "cat", "car", "motorcycle", "bus", "truck", "vehicle"]
        
        log.info(f"Processo Vision fallback recovery su video={len(video_to_suspects)}")

        pbar = tqdm(
            video_to_suspects.items(),
            desc="Vision Recovery",
            unit="vid",
            bar_format=f"{get_pbar_prefix('YOLO Scan')} {{rate_fmt:>10}} [{{bar}}] {{percentage:3.0f}}% {{n_fmt}}/{{total_fmt}} {{elapsed}}<{{remaining}}"
        )
        
        for v_name, suspects in pbar:
            best_report = None
            current_priority_idx = len(priority_order)

            for suspect in suspects:
                report = self.vision_engine.refine_fallback(suspect)
                
                if not report:
                    continue
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
                tqdm.write(f"✨ {Fore.GREEN}RECUPERATO:{Style.RESET_ALL} {v_name} -> {best_report.get('category')}")

        pbar.close()

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
            )

            log.info("Vision AI ready.")
            
        except ImportError as e:
            log.error(f"Error on setting up vision_engine, error={e}")
            self.is_refine = False

    def check_cameras_clean(self):
        """
        Confronta l'immagine di riferimento in /checks con una delle immagini NVR 
        già presenti nel full_index (fascia 00:00 - 05:00).
        """
        log.info("Lens Health Check Start")
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
                _, timestamp = parse_filename(
                    rec["video_path"], 
                    self.settings["storage_settings"]["filename_template"],
                    self.settings["storage_settings"]["timestamp_format"]
                )
                
                if timestamp and 0 <= timestamp.hour <= 5:
                    # Se il record ha delle immagini NVR associate, ne prendiamo una
                    if rec["nvr_images"]:
                        night_sample = rec["nvr_images"][0] 
                        break
            
            # 3. Se abbiamo entrambi i file, interroghiamo il Vision Engine
            if night_sample:
                log.info(f"Cam_id={cam_id}] comparison with image={Path(night_sample).name}")
             
                self._ensure_vision_initialized()
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
    
    def _get_reference_path(self, cam_id):
        """
        Cerca l'immagine di riferimento (es. 02.jpg) nella cartella 'checks'.
        """
        # Usiamo la costante CHECKS_DIR che punta alla cartella /checks
        for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG"]:
            potential_ref = CHECKS_DIR / f"{cam_id}{ext}"
            if potential_ref.exists():
                return potential_ref
        
        return None
    

    def _get_final_summary(self, total_time: float) -> str:
        stats = {}
        for res in self.final_reports:
            cat = res.get('category', 'unknown')
            stats[cat] = stats.get(cat, 0) + 1
        total_videos = len(self.final_reports)
        lines = [
            f"⏱️  Total_time={total_time:.2f}s",
            f"🎥 Processed num_video={total_videos}",
            "Category",
        ]
        for cat, count in stats.items():
            lines.append(f"  - category={cat.capitalize():<12} | count={count}")
        return "\n".join(lines)

    def _print_final_summary(self, total_time: float):
        log.info("-" * 50)
        log.info("           Results")
        log.info("-" * 50)
        for line in self._get_final_summary(total_time).split("\n"):
            log.info(line)
    log.info("-" * 50)

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
    
    def _check_engine_integrity(self):
        final_report_path = self.output_dir / FINAL_REPORT
        
        # Se non c'è il report finale, la cartella è considerata "nuova"
        if not final_report_path.exists():
            return True 

        # Definiamo i percorsi delle cache
        vision_cache_file = self.output_dir / VISION_CACHE
        blip_cache_file = self.output_dir / CLIPBLIP_CACHE

        # LOGICA DI BLOCCO
        # Se l'utente ha scelto BLIP ma esiste la cache di VISION
        if self.engine == "blip" and vision_cache_file.exists():
            log.critical("--- ENGINE MISMATCH DETECTED ---")
            log.critical(f"Reason: Found Vision AI cache ({VISION_CACHE}) but current engine is BLIP.")
            log.critical("Please clear output folder or change engine.")
            return False

        # Se l'utente ha scelto VISION ma esiste la cache di BLIP
        if self.engine == "vision" and blip_cache_file.exists():
            log.critical("--- ENGINE MISMATCH DETECTED ---")
            log.critical(f"Reason: Found BLIP cache ({CLIPBLIP_CACHE}) but current engine is VISION.")
            log.critical("Please clear output folder or change engine.")
            return False

        return True