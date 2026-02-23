import logging
from pathlib import Path
import shutil

from smart_surveillance_sorter.utils import get_safe_path
log = logging.getLogger(__name__) 

class FileSorter:
    def __init__(self, settings, input_dir, work_dir, is_test=False):
        
        self.settings = settings
        self.input_dir = Path(input_dir)
        self.work_dir = Path(work_dir)
        self.is_test = is_test
        
        # Determina la strategia
        self.inplace = self.input_dir in self.work_dir.parents or self.input_dir == self.work_dir
        self.method = "COPY" if (self.is_test or not self.inplace) else "MOVE"
        
        # Struttura cartelle dal settings
        storage_cfg = settings.get("storage_settings", {})
        self.structure_type = storage_cfg.get("structure_type", "camera_first")

    # def _execute_io(self, src, dst):
    #     """Esegue fisicamente lo spostamento o la copia."""
    #     src = Path(src)
    #     dst = Path(dst)
        
    #     if not src.exists():
    #         return False
            
    #     try:
    #         dst.parent.mkdir(parents=True, exist_ok=True)
    #         if self.method == "MOVE":
    #             shutil.move(str(src), str(dst))
    #         else:
    #             shutil.copy2(str(src), str(dst))
    #         return True
    #     except Exception as e:
    #         log.critical(f"Errore durante {self.method} di {src.name}: {e}")
    #         return False
        
    def _execute_io(self, src, dst):
        """Esegue fisicamente lo spostamento o la copia."""
        src = Path(src)
        dst = Path(dst)
        
        # 1. SE LA SORGENTE NON C'È: 
        # In caso di Resume con MOVE, il file è già sparito dall'input. 
        # Ritorniamo False e il Sorter passa oltre senza errori.
        if not src.exists():
            return False
            
        # 2. SE LA DESTINAZIONE C'È GIÀ:
        # In caso di Resume con COPY, non vogliamo riscrivere il file.
        if dst.exists():
            return True

        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if self.method == "MOVE":
                shutil.move(str(src), str(dst))
            else:
                shutil.copy2(str(src), str(dst))
            return True
        except Exception as e:
            # Usiamo log.error per non interrompere il ciclo su altri video
            log.error(f"Error on {self.method} of {src.name}.  error={e}")
            return False

    def _process_item(self, item, base_dir):
        """
        item: il record dal JSON finale (o dalla lista risultati)
        base_dir: la cartella di input o fallback
        """
        video_src = Path(item["video_path"])
        
        # 1. Determina la cartella di destinazione (es. Orto/person)
        target_dir = get_safe_path(
            base_dir, 
            item["camera_name"], 
            item["category"], 
            self.structure
        )
        target_dir.mkdir(parents=True, exist_ok=True)

        # 2. Sposta/Copia il VIDEO
        dest_video = target_dir / video_src.name
        self._execute_io(video_src, dest_video)

        # 3. Gestione dei FRAME (dalla tua lista 'frames')
        # Se abbiamo dei frame estratti per questo video, portiamoli con noi
        if "frames" in item and item["frames"]:
            # Creiamo una sottocartella per i frame
            frames_target_dir = target_dir / "frames" / video_src.stem
            frames_target_dir.mkdir(parents=True, exist_ok=True)
            
            for frame_info in item["frames"]:
                frame_src = Path(frame_info["frame_path"])
                if frame_src.exists():
                    dest_frame = frames_target_dir / frame_src.name
                    self._execute_io(frame_src, dest_frame)
                    
                # Se esiste anche il crop (molto utile!), spostiamo anche quello
                if "crop_path" in frame_info and frame_info["crop_path"]:
                    crop_src = Path(frame_info["crop_path"])
                    if crop_src.exists():
                        dest_crop = frames_target_dir / crop_src.name
                        self._execute_io(crop_src, dest_crop)

   
    def sort_all(self, final_results, raw_results, full_index):
        processed_videos = {}
        processed_files = set() # <--- IL NOSTRO VIGILE URBANO
      
        # 1. CONTROLLO PERMESSI TATTICO
        import os
        if os.access(self.input_dir, os.W_OK):
            # Caso normale: scriviamo nell'input
            self.dest_base = self.input_dir
            log.info(f"Dir={self.input_dir}")
        else:
            # Caso emergenza: scriviamo nella work_dir (che è temp_dir)
            self.dest_base = self.work_dir / "SMI_SORTED_RESULTS"
            self.dest_base.mkdir(parents=True, exist_ok=True)
            log.warning(f"Input is not writeable! Risults on folder={self.dest_base}")
        # --- CARICAMENTO MAPPING REALE ---
        # Usiamo la tua funzione che legge cameras.json, non settings.json!
        from utils import get_camera_mapping 
        camera_mapping = get_camera_mapping() 
        
        #log.debug(f"--- DEBUG: Mapping caricato da cameras.json: {camera_mapping}")


        classificati_paths = {item["video_path"] for item in final_results}

        # 2. Integriamo i video mancanti da full_index
        for cam_id, entries in full_index.items():
            cam_id_str = str(cam_id).zfill(2) if str(cam_id).isdigit() else str(cam_id)
            cam_name = camera_mapping.get(cam_id_str, f"camera_{cam_id_str}")
            
            for entry in entries:
                v_path = str(entry["video_path"])
                if v_path not in classificati_paths:
                    # Lo aggiungiamo come 'others'
                    final_results.append({
                        "camera_id": cam_id_str,
                        "camera_name": cam_name,
                        "video_name": Path(v_path).name,
                        "video_path": v_path,
                        "category": "others",
                        "confidence": 0,
                        "best_frame_path": None,
                        "engine": "none"
                    })
                    classificati_paths.add(v_path)

        files_processati = set() # Per evitare di copiare due volte la stessa immagine NVR

        for item in final_results:
            v_path = item["video_path"]
            cam_name = item["camera_name"]
            #cat = item["category"]
            
            # --- NORMALIZZAZIONE CATEGORIA ---
            # Se l'IA ha restituito 'nothing', forziamo in 'others'
            cat = item["category"]
            if cat == "nothing":
                cat = "others"
            target_dir = get_safe_path(self.dest_base, cam_name, cat, self.structure_type)
            
            # 1. SPOSTA VIDEO
            if v_path not in files_processati:
                if self._execute_io(v_path, target_dir / item["video_name"]):
                    files_processati.add(v_path)

        
            video_details = next((r for r in raw_results if r["video_path"] == v_path), None)
            if video_details and "frames" in video_details:
                for f_info in video_details["frames"]:
                    
                    # Frame originale
                    f_src = f_info.get("frame_path")
                    if f_src and f_src not in files_processati:
                        if self._execute_io(f_src, target_dir / Path(f_src).name):
                            files_processati.add(f_src)
                    
                    # Immagine Crop (quella zoomata per Qwen)
                    c_src = f_info.get("crop_path")
                    if c_src and c_src not in files_processati:
                        if self._execute_io(c_src, target_dir / Path(c_src).name):
                            files_processati.add(c_src)        

            # 3. SPOSTA IMMAGINI NVR (Cercandole nel full_index)
            # Troviamo il record nel full_index per questo video
            for cam_id, entries in full_index.items():
                entry = next((e for e in entries if str(e["video_path"]) == v_path), None)
                if entry and "nvr_images" in entry:
                    for img_p in entry["nvr_images"]:
                        img_src = str(img_p)
                        if img_src not in files_processati:
                            if self._execute_io(img_src, target_dir / Path(img_src).name):
                                files_processati.add(img_src)

        

    def cleanup(self):
        """
        Rimuove l'intera cartella cache se siamo in modalità MOVE.
        """
        if self.method == "MOVE" and self.work_dir.exists():
            # Controllo di sicurezza: non cancellare la root dell'utente!
            if self.work_dir.resolve() == self.input_dir.resolve():
                log.error("🛑 Dir protection active. output_dir is input_dir. Cleanup abort.")
                return

            import shutil
            try:
                shutil.rmtree(self.work_dir)
                log.info(f"Cache folder removed. Folder={self.work_dir.name}")
            except Exception as e:
                log.error(f"Error removing cache: error={e}")