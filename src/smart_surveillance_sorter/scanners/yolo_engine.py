from datetime import datetime
import json
import logging
from pathlib import Path
import shutil
from typing import Optional

import cv2
from smart_surveillance_sorter.models import load_smart_yolo
from smart_surveillance_sorter.utils import  get_crop_coordinates, get_target_ids, get_video_capture
from .yolo_helpers import extract_frames_with_cache
log = logging.getLogger(__name__) 
class YoloEngine:
    """
    YOLO inference engine for fast, lightweight object detection on NVR images
    and video frames.

    The engine loads a smart YOLO model (default `yolov8l`) onto the specified
    device and provides properties for accessing its settings and camera
    configuration. It offers a quick‑scan routine for single images
    (`scan_single_image`) that applies a minimum confidence threshold of
    0.60 and an area‑ratio filter to avoid tiny glitches or distant birds
    [1]. When no valid target is found during a fallback scan, the method
    logs a debug message and returns `None` [1].
    """
    def __init__(self,mode,device,settings,cameras_config):
      
        self.mode   =   mode
        self.device =   device
        

        #file di configurazione
        self.settings          =  settings
        self.cameras_config    =  cameras_config
        yolo_cfg                = self.settings.get("yolo_settings", {})

        # carica il modello yolo
        self.model = load_smart_yolo(
            model_name=yolo_cfg.get("model_path", "yolov8l"), 
            device=self.device
        )

        self.name_to_id = {v: k for k, v in self.model.names.items()}

    
    
    def scan_single_image(self, image_path: Path, video_path: Path,  frames_dir: Path,cam_id: str):
            """
            Run a YOLO‑based detection on a single image extracted from a video.

            Parameters
            ----------
            image_path : Path
                Path to the image file that was extracted from the video.
            video_path : Path
                Path to the original video file from which the image was taken.
            cam_id : str
                Identifier of the camera that captured the video.

            Returns
            -------
            Optional[dict]
                If a detection is found, returns a dictionary containing the
                following keys:
                - ``camera_id``: the camera identifier,
                - ``video_name``: the name of the video file,
                - ``video_path``: the full path to the video,
                - ``image_path``: the full path to the image,
                - ``yolo_category``: the detected category,
                - ``yolo_data``: the raw YOLO detection data.
                If no detection is found, returns ``None``.

            Notes
            -----
            The method prepares the data for the Scanner module by
            packaging the best YOLO detection found in the image. The
            ``video_name`` is extracted from ``video_path`` for the final
            report. If the detection confidence is below a threshold, the
            method simply returns ``None``. The timestamp of the image is
            typically derived from the file’s modification time. [1]
            """
            # min_confidence = 0.60
            # min_area_ratio = 0.002 # Per evitare piccoli glitch/uccelli lontano

            # # YOLO v8 può leggere direttamente dal path (più veloce di cv2.imread se non serve pre-processare)
            # res = self.model(str(image_path), classes=[0], verbose=False,device=self.device)[0]

            # best_det = None
            # max_conf = 0
            
            # # Area totale per il filtro dimensione
            # h, w = res.orig_shape
            # img_area = h * w

            # for box in res.boxes:
            #     conf = float(box.conf[0])
                
            #     if conf < min_confidence:
            #         continue

            #     x1, y1, x2, y2 = [int(x) for x in box.xyxy[0]]
            #     area = (x2 - x1) * (y2 - y1)
                
            #     if area / img_area < min_area_ratio:
            #         continue

            #     if conf > max_conf:
            #         max_conf = conf
            #         best_det = {
            #             "confidence": conf,
            #             "bbox": [x1, y1, x2, y2],
            #         }

            # if not best_det:
            #     return None

            # # Usiamo la data di modifica del file come timestamp se non diversamente specificato
            # ts = datetime.fromtimestamp(image_path.stat().st_mtime)

            # return {
            #     "camera_id": cam_id,
            #     "video_path": str(video_path),
            #     "categories_found": ["person"],
            #     "frames": [
            #         {
            #             "category": "person",
            #             "frame_path": str(image_path),
            #             "confidence": best_det["confidence"],
            #             "bbox": best_det["bbox"],
            #             "timestamp": ts.isoformat()
            #         }
            #     ],
            #     "resolved_by": "nvr_image"
            # }

            #def scan_single_image(self, image_path: Path, video_path: Path, cam_id: str):
            min_confidence = 0.60
            min_area_ratio = 0.002 

            res = self.model(str(image_path), classes=[0], verbose=False, device=self.device)[0]
            class_names = res.names
            best_det = None
            max_conf = 0
            
            h, w = res.orig_shape
            img_area = h * w

            for box in res.boxes:
                conf = float(box.conf[0])

                if conf < min_confidence:
                    continue

                x1, y1, x2, y2 = [int(x) for x in box.xyxy[0]]
                area = (x2 - x1) * (y2 - y1)
                
                if area / img_area < min_area_ratio:
                    continue

                if conf > max_conf:
                    max_conf = conf
                   # Recuperiamo l'ID e il nome della classe
                    cls_id = int(box.cls[0])
                    label_name = class_names.get(cls_id, "unknown")

                    # Salviamo TUTTO dentro best_det
                    best_det = {
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2],
                        "label": label_name # <--- IMPORTANTE: aggiungilo qui
                    }

                    # Ora creiamo il log usando i dati appena salvati
                    yolo_log_path = Path("yolo_detailed_log.jsonl")
                    yolo_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "video": str(video_path.name),
                        "camera": cam_id,
                        "class_detected": label_name, # Usa direttamente label_name
                        "confidence": round(conf, 4),
                        "bbox": [x1, y1, x2, y2]
                    }
            
                    with open(yolo_log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(yolo_entry) + "\n")
            # -------------------------

            if not best_det:
                return None

            # --- LOGICA DI CROP AGGIUNTA ---
            # # Generiamo il path per il crop (es: nomefile_crop.jpg)
            # crop_filename = f"{image_path.stem}_crop{image_path.suffix}"
            # crop_path = image_path.parent / crop_filename

          # --- LOGICA DI COPIA E CROP NVR ---
            self.frames_dir = frames_dir
            self.frames_dir.mkdir(parents=True, exist_ok=True)

            # 1. Definiamo i nuovi percorsi dentro 'extracted_frames'
            nvr_original_dest = self.frames_dir / image_path.name
            crop_filename = f"{image_path.stem}_crop{image_path.suffix}"
            crop_path = self.frames_dir / crop_filename
            
            try:
                # 2. Copiamo prima l'immagine originale NVR nella cartella dei frame
                shutil.copy2(image_path, nvr_original_dest)
                # Carichiamo l'immagine originale
                img = cv2.imread(str(image_path))
                if img is not None:
                    # Calcoliamo le coordinate con il tuo metodo (usiamo margin_perc=1.0 per dare respiro)
                    cx1, cy1, cx2, cy2 = get_crop_coordinates(best_det["bbox"], img.shape)
                    
                    # Tagliamo e salviamo
                    crop_img = img[cy1:cy2, cx1:cx2]
                    cv2.imwrite(str(crop_path), crop_img)
                    best_det["crop_path"] = str(crop_path)
                else:
                    best_det["crop_path"] = None
            except Exception as e:
                log.critical(f"⚠️ Errore durante il ritaglio dell'immagine NVR: error={e}")
                best_det["crop_path"] = None
            # ------------------------------

            ts = datetime.fromtimestamp(image_path.stat().st_mtime)

            return {
                "camera_id": cam_id,
                "video_path": str(video_path),
                "categories_found": ["person"],
                "frames": [
                    {
                        "category": "person",
                        #"frame_path": str(image_path),
                        "frame_path": str(nvr_original_dest),
                        "crop_path": best_det.get("crop_path"), # <--- Passiamo il crop!
                        "confidence": best_det["confidence"],
                        "bbox": best_det["bbox"],
                        "timestamp": ts.isoformat()
                    }
                ],
                "resolved_by": "nvr_image"
            }
    
    # def scan_video(self, video_path: Path, frames_dir: Path, cam_id: str) -> Optional[dict]:
      
    #     # ------------------------------------------------------------------
    #     # 1️⃣ Configurazione e Target IDs
    #     # ------------------------------------------------------------------
    #     cam_cfg = self.cameras_config.get(str(cam_id), {})
    #     ignore_labels = cam_cfg.get("filters", {}).get("ignore_labels", [])
    #     self.frames_dir=frames_dir
    #     final_target_ids = get_target_ids(
    #         model=self.model, 
    #         settings=self.settings, 
    #         mode=self.mode, 
    #         camera_ignore_labels=ignore_labels
    #     )

    #     if not final_target_ids:
    #         log.warning(f"Nessuna classe target per cam {cam_id}")
    #         return None

    #     # ------------------------------------------------------------------
    #     # 2️⃣ Apertura Video e Parametri Temporali
    #     # ------------------------------------------------------------------
    #     cap = get_video_capture(video_path)
    #     if cap is None: 
    #         return None

    #     fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    #     stride = self.settings["yolo_settings"].get("vid_stride", 12)
        
    #     time_gap_sec = self.settings["yolo_settings"].get("time_gap_sec", 3)
    #     occ_gap_frames = int(fps * time_gap_sec)
    #     num_occurrence = self.settings["yolo_settings"].get("num_occurrence", 3)
        
    #     yolo_override_threshold = self.settings["yolo_settings"].get("yolo_override_threshold")
    #     # ------------------------------------------------------------------
    #     # 3️⃣ Ottimizzazione Costanti (Fuori dai Loop)
    #     # ------------------------------------------------------------------
    #     yolo_set = self.settings.get("yolo_settings", {})
    #     thresholds = yolo_set.get("thresholds", {})
    #     detection_groups = yolo_set.get("detection_groups", {})

    #     # Creiamo una mappa veloce: {"dog": "animal", "person": "person", ...}
    #     label_to_group = {}
    #     for group_name, labels in detection_groups.items():
    #         g_key = group_name.lower()
    #         for lbl in labels:
    #             label_to_group[lbl] = g_key

    #     # Inizializzazione Tracking
    #     detections = {g.lower(): [] for g in detection_groups.keys()}
    #     occ_count = {g.lower(): 0 for g in detection_groups.keys()}
    #     last_saved_frame = {g.lower(): -occ_gap_frames for g in detection_groups.keys()}

    #     prev_stride = stride
    #     frame_idx = 0
    #     self.last_detection_idx = -9999  # Inizializzazione per il cooldown
        
    #     current_stride = stride
    #     skip_counter = current_stride  # Questo controllerà i salti
    #     # ------------------------------------------------------------------
    #     # 4️⃣ Loop di Analisi
    #     # ------------------------------------------------------------------
    #     while True:

    #         # --- LOGICA DEL SALTO ---
    #         # Se non siamo al frame giusto, facciamo un grab() veloce e saltiamo tutto
    #         if skip_counter < current_stride:
    #             if not cap.grab(): # Sposta il puntatore senza decodificare
    #                 break
    #             frame_idx += 1
    #             skip_counter += 1
    #             continue
    #         ret, frame = cap.read() # Leggiamo sempre il frame successivo (sequenziale = veloce)
    #         if not ret:
    #             break 

    #         frame_idx += 1
    #         skip_counter += 1

    #         # Analizziamo solo se il contatore ha raggiunto lo stride deciso
    #         # if skip_counter < current_stride:
    #         #     continue
    #         found_valid_target = False
    #         yolo_results = self.model.predict(
    #             source=frame, 
    #             classes=final_target_ids, 
    #             verbose=False, 
    #             conf=0.20,
    #             device=self.device 
    #         )[0]

            
    #         skip_counter = 0  # Reset del cont
    #         if len(yolo_results.boxes) == 0:
    #             continue

    #         for box in yolo_results.boxes:
    #             cls_id = int(box.cls[0])
    #             conf = float(box.conf[0])
    #             label = self.model.names[cls_id]
                
    #             # 1. Recupero Gruppo (es. "person", "animal", "vehicle")
    #             group = label_to_group.get(label)
    #             if not group:
    #                 continue

    #             # 2. Controllo Confidenza dal JSON
    #             if conf < thresholds.get(group, 0.50):
    #                 continue
                
    #             found_valid_target = True
    #             self.last_detection_idx = frame_idx

    #             # 3. Calcolo Area (come nel tuo codice vecchio)
    #             x1, y1, x2, y2 = map(int, box.xyxy[0])
    #             bbox_area = (x2 - x1) * (y2 - y1)
    #             h, w = frame.shape[:2]
    #             area_ratio = bbox_area / (h * w)

    #             # 4. SALVATAGGIO SEMPRE (Se sopra soglia, lo mettiamo in lista)
    #             detections[group].append({
    #                 "frame_idx": frame_idx,
    #                 "confidence": conf,
    #                 "bbox": [x1, y1, x2, y2],
    #                 "area": bbox_area,
    #                 "area_ratio": area_ratio,
    #                 "label": label,
    #                 "yolo_reliable": conf >= yolo_override_threshold
    #             })

    #             # 5. LOGICA EARLY-STOP (Solo per persone e animali, con gap temporale)
    #             if group in ["person", "animal"]:
    #                 if frame_idx - last_saved_frame[group] >= occ_gap_frames:
    #                     last_saved_frame[group] = frame_idx
    #                     occ_count[group] += 1
            
    #         # --- FINE LOOP BOX ---

    #         # 6. Controllo Uscita Anticipata
    #         # Se abbiamo raggiunto il numero di occorrenze distinte per persone o animali
    #         if occ_count["person"] >= num_occurrence or occ_count["animal"] >= num_occurrence:
    #             log.info(f"Early stop su {video_path.name}: raggiunte occorrenze target")
    #             break
            
    #         # --- QUI VA IL CALCOLO DELLO STRIDE ---
    #         # Fuori dal for box, ma dentro il while True
    #         current_stride = self._get_next_stride(
    #             frame_idx, 
    #             fps, 
    #             found_valid_target, 
    #             cam_cfg
    #         )
    #         # --- LOG DI DEBUG ---
    #         if current_stride != prev_stride:
    #             fase = "PROTEZIONE/DETECTION" if current_stride == 12 else "VELOCITÀ CROCIERA"
    #             print(f"[{video_path.name}] Frame {frame_idx}: Cambio stride a {current_stride} ({fase})")
    #             prev_stride = current_stride
    # # --------------------
    #         # Aggiorniamo l'indice per il prossimo ciclo
    #         frame_idx += current_stride 

    #     # --- FINE WHILE VIDEO ---
    #     cap.release()
    #     if not any(detections.values()):
    #         return None

    #     # salvataggio frame
    #     self.frames_dir.mkdir(parents=True, exist_ok=True)
    #     cap = cv2.VideoCapture(str(video_path))
    #     saved_frames = extract_frames_with_cache(
    #             cap=cap,
    #             detections=detections,
    #             fps=fps,
    #             video_path=video_path,
    #             frames_dir=self.frames_dir,
    #             frames_per_category=num_occurrence
    #         )
    #     cap.release()


    #     if not saved_frames:
    #         return None

    #     # Troviamo solo le categorie che hanno almeno una rilevazione salvata
    #     categories_found = [cat for cat, found_list in detections.items() if len(found_list) > 0]

    #     return {
    #         "camera_id": cam_id,
    #         "video_path": str(video_path),
    #         "categories_found": categories_found,
    #         "frames": saved_frames,  # Questa è la lista prodotta da extract_frames_with_cache
    #         "timestamp": datetime.now().isoformat()
    #     }
    
    def scan_video(self, video_path: Path, frames_dir: Path, cam_id: str) -> Optional[dict]:
        # ------------------------------------------------------------------
        # 1️⃣ Configurazione e Target IDs
        # ------------------------------------------------------------------
        cam_cfg = self.cameras_config.get(str(cam_id), {})
        ignore_labels = cam_cfg.get("filters", {}).get("ignore_labels", [])
        thresholds = cam_cfg.get("thresholds", {})
        self.frames_dir = frames_dir
        
        final_target_ids = get_target_ids(
            model=self.model, 
            settings=self.settings, 
            mode=self.mode, 
            camera_ignore_labels=ignore_labels
        )

        if not final_target_ids:
            log.warning(f"Nessuna classe target per cam={cam_id}")
            return None

        # ------------------------------------------------------------------
        # 2️⃣ Apertura Video e Parametri Temporali
        # ------------------------------------------------------------------
        cap = get_video_capture(video_path)
        if cap is None: 
            return None

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        stride_std = self.settings["yolo_settings"].get("vid_stride", 12)
        
        time_gap_sec = self.settings["yolo_settings"].get("time_gap_sec", 3)
        occ_gap_frames = int(fps * time_gap_sec)
        num_occurrence = self.settings["yolo_settings"].get("num_occurrence", 3)
        yolo_override_threshold = self.settings["yolo_settings"].get("yolo_override_threshold", 0.85)

        # ------------------------------------------------------------------
        # 3️⃣ Ottimizzazione Costanti e Tracking
        # ------------------------------------------------------------------
        yolo_set = self.settings.get("yolo_settings", {})
        #thresholds = yolo_set.get("thresholds", {})
        detection_groups = yolo_set.get("detection_groups", {})

        label_to_group = {}
        for group_name, labels in detection_groups.items():
            g_key = group_name.lower()
            for lbl in labels:
                label_to_group[lbl] = g_key

        detections = {g.lower(): [] for g in detection_groups.keys()}
        occ_count = {g.lower(): 0 for g in detection_groups.keys()}
        last_saved_frame = {g.lower(): -occ_gap_frames for g in detection_groups.keys()}

        # Variabili di controllo Stride
        frame_idx = 0
        self.last_detection_idx = -9999
        current_stride = stride_std
        prev_stride = stride_std

        #log.debug(f"Video={video_path.name} ha FPS={fps}")
        
        # ------------------------------------------------------------------
        # 4️⃣ Loop di Analisi
        # ------------------------------------------------------------------
        while True:
            # LEGGIAMO IL FRAME CORRENTE
            ret, frame = cap.read()
            if not ret:
                break
            
            # Siamo sul frame 'frame_idx'
            found_valid_target = False

            # ANALISI YOLO
            yolo_results = self.model.predict(
                source=frame, 
                classes=final_target_ids, 
                verbose=False, 
                conf=0.20,
                device=self.device 
            )[0]

            if len(yolo_results.boxes) > 0:
                for box in yolo_results.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = self.model.names[cls_id]
                    group = label_to_group.get(label)
                    
                    if not group or conf < thresholds.get(group, 0.50):
                        continue
                    
                    # Detection Valida
                    found_valid_target = True
                    self.last_detection_idx = frame_idx

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    bbox_area = (x2 - x1) * (y2 - y1)
                    h, w = frame.shape[:2]
                    area_ratio = bbox_area / (h * w)

                    detections[group].append({
                        "frame_idx": frame_idx,
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2],
                        "area": bbox_area,
                        "area_ratio": area_ratio,
                        "label": label,
                        "yolo_reliable": conf >= yolo_override_threshold
                    })

                    if group in ["person", "animal"]:
                        if frame_idx - last_saved_frame[group] >= occ_gap_frames:
                            last_saved_frame[group] = frame_idx
                            occ_count[group] += 1

            # Controllo Early Stop
            if occ_count["person"] >= num_occurrence or occ_count["animal"] >= num_occurrence:
                log.debug(f"Early stop su video={video_path.name}")
                break
            
            # CALCOLO STRIDE PER IL PROSSIMO SALTO
            current_stride = self._get_next_stride(frame_idx, fps, found_valid_target, cam_cfg)

            if current_stride != prev_stride:
                fase = "PROTEZIONE/DETECTION" if current_stride <= stride_std else "VELOCITÀ CROCIERA"
                #print(f"[{video_path.name}] Frame {frame_idx}: Cambio stride a {current_stride} ({fase})")
                prev_stride = current_stride

            # --- LOGICA DEL SALTO (Skip) ---
            # Abbiamo già letto 1 frame con cap.read(), ne saltiamo (current_stride - 1)
            if current_stride > 1:
                for _ in range(current_stride - 1):
                    if not cap.grab(): # Grab è veloce, non decodifica
                        break
                    frame_idx += 1
            
            # Incrementiamo l'indice del frame letto con read()
            frame_idx += 1

        cap.release()

        # ------------------------------------------------------------------
        # 5️⃣ Estrazione Frame Finali (Solo se ci sono detection)
        # ------------------------------------------------------------------
        if not any(detections.values()):
            return None

        self.frames_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        saved_frames = extract_frames_with_cache(
                cap=cap,
                detections=detections,
                fps=fps,
                video_path=video_path,
                frames_dir=self.frames_dir,
                frames_per_category=num_occurrence
            )
        cap.release()

        if not saved_frames:
            return None

        categories_found = [cat for cat, found_list in detections.items() if len(found_list) > 0]

        return {
            "camera_id": cam_id,
            "video_path": str(video_path),
            "categories_found": categories_found,
            "frames": saved_frames,
            "timestamp": datetime.now().isoformat()
        }
    def _get_next_stride(self, frame_idx, fps, found_valid_target, cam_config):
        """
        Calcola il salto di frame (stride) per il ciclo YOLO.
        """
        # 1. Se la telecamera non ha lo stride dinamico, usa lo standard dai settings
        if not cam_config.get("dynamic_stride", False):
            return self.settings["yolo_settings"].get("vid_stride", 12)

        # 2. Recupero parametri dai settings (con fallback di sicurezza)
        dyn_settings = self.settings["yolo_settings"].get("dynamic_stride_settings", {})
        stride_std = self.settings["yolo_settings"].get("vid_stride", 12)
        stride_fast = dyn_settings.get("stride_fast", 25)
        pre_roll_sec = dyn_settings.get("pre_roll_sec", 25)
        cooldown_sec = dyn_settings.get("cooldown_sec", 5)

        # 3. LOGICA DI STATO
        
        # A. Se abbiamo un rilevamento attivo, siamo in "Attention Mode"
        if found_valid_target:
            # Aggiorniamo il timestamp dell'ultimo avvistamento (nel loop principale)
            return stride_std
        
        # B. Se siamo nel Pre-Roll (zona protetta iniziale)
        if frame_idx < (pre_roll_sec * fps):
            return stride_std

        # C. Se siamo nel Cooldown (abbiamo visto qualcosa poco fa)
        # Calcoliamo quanti frame sono passati dall'ultima detection
        frames_since_last = frame_idx - self.last_detection_idx
        if frames_since_last < (cooldown_sec * fps):
            return stride_std

        # D. In tutti gli altri casi: Velocità Crociera
        return stride_fast
    def scan_fallback(self, image_path, target_ids):
        """
        Perform a high‑sensitivity scan on a single low‑confidence NVR image.

        The Vision engine applies a very sensitive detector to a single
        image and returns any detections that exceed a predefined
        confidence threshold.  The returned detection is used by the
        scanner to attempt recovery of events that were missed during
        the initial YOLO pass.

        Parameters
        ----------
        image_path : str
            Path to the NVR image to be analyzed.
        target_ids : list[int]
            Class IDs that the Vision engine should look for.

        Returns
        -------
        dict or None
            A report dictionary containing detection details (e.g.,
            ``category``) if a confident detection is found; otherwise
            ``None``.  The scanner stops scanning further images for the
            current video once a detection is returned. [1]
        """
        # Confidenza bassissima: 0.1 o 0.15
        res = self.model(str(image_path), classes=target_ids, conf=0.15, verbose=False,device=self.device)[0]
        
        if len(res.boxes) > 0:
            # Prendiamo la migliore anche se debole
            best_box = res.boxes[0]
            return {
                "confidence": float(best_box.conf[0]),
                "bbox": [int(x) for x in best_box.xyxy[0]],
                "yolo_reliable": False # Segnaliamo che è un sospetto
            }
        return None
    
    def run_fallback_phase(self, current_results):
        """
        Execute the fallback detection phase for videos that were not identified
        during the primary YOLO scan.

        The method logs the start of the fallback process, determines the
        fallback targets from *current_results*, and, if any targets exist,
        runs a low‑confidence detection routine on the relevant frames.
        New detections are collected and appended to *current_results*.

        Parameters
        ----------
        current_results : list[dict]
            A list of result dictionaries produced by the main scan. Each
            dictionary must contain at least a ``camera_id`` and a list of
            detected categories.

        Returns
        -------
        list[dict]
            The original *current_results* augmented with any detections that
            were found during the fallback phase. If no fallback targets are
            present, the original list is returned unchanged.

        Notes
        -----
        - The method logs the beginning of the fallback phase with a
        stylized detective emoji to aid debugging.
        - Targets for fallback are derived from ``_get_fallback_targets``; only
        videos lacking a reliable detection are processed.
        - The fallback routine uses a very low confidence threshold to capture
        subtle shadows or distant objects, as described in the engine’s
        documentation [1].
        """
        log.info("🕵️ Inizio fase FALLBACK per video non identificati...")
        
        fallback_targets = self._get_fallback_targets(current_results)
        if not fallback_targets:
            return current_results

        #new_detections = []
        for item in fallback_targets:
            # 1. YOLO controlla le immagini NVR a bassa confidenza
            suspicious_frames = []
            target_ids = self.yolo_engine.get_target_ids(item["camera_id"]) # Helper
            
            for img_path in item["nvr_images"]:
                detection = self.yolo_engine.scan_fallback(img_path, target_ids)
                if detection:
                    # Creiamo un mini-report per la Vision AI
                    suspicious_frames.append({
                        "video_name": item["video_name"],
                        "video_path": item["video_path"],
                        "camera_id": item["camera_id"],
                        "frame_path": img_path,
                        "yolo_data": detection
                    })
            
            # 2. Se YOLO ha dei sospetti, mandiamo solo quelli alla Vision AI
            for suspect in suspicious_frames:
                log.debug(f"🔍 Fallback: Vision AI controlla sospetto in video={suspect['video_name']}")
                
                # Chiamiamo la Vision AI (usiamo il motore che abbiamo già)
                report = self.vision_engine.refine_single_video(suspect)
                
                # Se la Vision AI conferma (es. category != "nothing")
                if report and report.get("category") != "nothing":
                    log.debug(f"✨ RECUPERATO: video={item['video_name']} classificato come category={report['category']}")
                    current_results.append(report)
                    break # Trovato uno, passiamo al prossimo video

        return current_results
    
    def low_conf_image_scan(self, image_path, video_path, cam_id):
        """
        Perform a very high‑sensitivity analysis on a single image.

        This method is used by the scanner when a low‑confidence
        detection from the YOLO model needs to be re‑evaluated on the
        corresponding NVR image with increased sensitivity.  It
        accepts the path to the image, the original video path, and
        the camera ID, then returns a detailed report.

        Parameters
        ----------
        image_path : str
            Path to the NVR image to analyze.
        video_path : str
            Path to the original video that produced the low‑confidence
            detection.
        cam_id : str
            Identifier of the camera that captured the video.

        Returns
        -------
        dict
            A report dictionary with the detected category and any
            additional metadata.

        Notes
        -----
        The method is designed to be highly sensitive; it may produce
        false positives but helps recover true events that were missed
        earlier. [1]
        """
        cam_cfg = self.cameras_config.get(str(cam_id), {})
        ignore_labels = cam_cfg.get("filters", {}).get("ignore_labels", [])
        
        target_ids = get_target_ids(
            model=self.model,
            settings=self.settings,
            mode=self.mode,
            camera_ignore_labels=ignore_labels
        )

        # Eseguiamo YOLO con confidenza molto bassa (es. 0.15)
        results = self.model(str(image_path), classes=target_ids, conf=0.05, verbose=False,device=self.device)
        
        # if not results or len(results[0].boxes) == 0:
        #     return None
        
        if not results or len(results[0].boxes) == 0:
            # Aggiungi questo per il debug
            log.debug(f"🔍 YOLO Fallback: Nessun target trovato in img={image_path.name}")
            return None

        # Prendiamo la detenzione con confidenza maggiore tra quelle trovate
        best_det = None
        max_conf = -1
        
        for box in results[0].boxes:
            conf = float(box.conf[0])
            #class_names = box.names
            if conf > max_conf:
                max_conf = conf
                cls_id = int(box.cls[0])
                # Prendi il nome della classe direttamente dal modello
                label_name = self.model.names.get(cls_id, "unknown")
                #label_name = class_names.get(cls_id, "unknown")
                label = self.model.names[cls_id]
                best_det = {
                    "category": label.lower(),
                    "confidence": conf,
                    "bbox": [int(x) for x in box.xyxy[0]]
                }

        # Se abbiamo un sospetto, prepariamo il dizionario per lo Scanner
        if best_det:
             # Ora creiamo il log usando i dati appena salvati

            yolo_log_path = Path("yolo_detailed_log.jsonl")
            yolo_entry = {
            "timestamp": datetime.now().isoformat(),
            "video": str(video_path.name),
            "camera": cam_id,
            "class_detected": label_name, # Usa direttamente label_name
            "confidence": round(conf, 4),
            "bbox": [int(x) for x in box.xyxy[0]]
            }
            with open(yolo_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(yolo_entry) + "\n") 
            return {
                "camera_id": cam_id,
                "video_name": Path(video_path).name, # Ci serve per il report finale
                "video_path": video_path,
                "image_path": image_path,
                "yolo_category": best_det["category"],
                "yolo_data": best_det
            }
        return None