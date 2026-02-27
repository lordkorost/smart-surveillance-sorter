"""
YoloEngine — Motore di inferenza YOLO per sorveglianza smart
=============================================================

Funzionalità principali:
  - Lazy loading del modello (carica solo quando serve)
  - scan_single_image: analisi immagini NVR ad alta confidenza (>=0.60)
  - scan_video: analisi video con stride adattivo e early exit
  - Soglie giorno/notte per telecamera (thresholds / thresholds_night)
  - Stride basato su FPS reali del video (vid_stride_sec in settings)
  - Warmup iniziale a stride ridotto (primi warmup_sec secondi)
  - Dynamic stride per telecamere con molto silenzio (es. parcheggio)
  - low_conf_image_scan: scan a bassa confidenza per fallback
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2

from smart_surveillance_sorter.models import load_smart_yolo
from smart_surveillance_sorter.utils import (
    get_crop_coordinates,
    get_target_ids,
    get_video_capture,
    get_smart_coordinates,
    is_night_astronomic,
)
from .yolo_helpers import extract_frames_with_cache

log = logging.getLogger(__name__)


class YoloEngine:
    """
    Motore YOLO con lazy loading, stride adattivo e soglie giorno/notte.
    """

    def __init__(self, mode, device, settings, cameras_config):
        self.mode           = mode
        self.device         = device
        self.settings       = settings
        self.cameras_config = cameras_config

        # Lazy loading — il modello viene caricato solo alla prima chiamata
        self.model      = None
        self.name_to_id = None
        self.yolo_cfg   = self.settings.get("yolo_settings", {})

        # Coordinate per calcolo astronomico notte (usate per soglie giorno/notte)
        city = self.settings.get("city", "Roma")
        self.lat, self.lon = get_smart_coordinates(city)

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def ensure_model_loaded(self):
        """Carica il modello in VRAM solo se non è già presente."""
        if self.model is None:
            #log.info("[Lazy Loading] YOLO Engine ...")
            self.model = load_smart_yolo(
                model_name=self.yolo_cfg.get("model_path", "yolov8l"),
                device=self.device,
            )
            self.name_to_id = {v: k for k, v in self.model.names.items()}

    # ------------------------------------------------------------------
    # Helper: soglie giorno/notte per telecamera
    # ------------------------------------------------------------------

    def _get_thresholds(self, cam_cfg: dict, timestamp: datetime) -> dict:
        """
        Restituisce le soglie corrette in base all'orario.
        Se la telecamera non ha 'thresholds_night', usa le soglie di giorno.
        """
        thresholds_day   = cam_cfg.get("thresholds", {})
        thresholds_night = cam_cfg.get("thresholds_night", thresholds_day)
        is_night = is_night_astronomic(timestamp, self.lat, self.lon)
        return thresholds_night if is_night else thresholds_day

    # ------------------------------------------------------------------
    # Scan immagine NVR (alta confidenza)
    # ------------------------------------------------------------------

    def scan_single_image(self, image_path: Path, video_path: Path,
                          frames_dir: Path, cam_id: str) -> Optional[dict]:
        """
        Analizza una singola immagine NVR con soglia alta (>=0.60).
        Usato per il fast-track: se l'immagine NVR è già sufficiente,
        il video viene classificato senza aprirlo.
        """
        MIN_CONFIDENCE = 0.60
        MIN_AREA_RATIO = 0.002

        self.ensure_model_loaded()

        res = self.model(str(image_path), classes=[0], verbose=False, device=self.device)[0]
        h, w = res.orig_shape
        img_area = h * w

        best_det = None
        max_conf  = 0.0

        for box in res.boxes:
            conf = float(box.conf[0])
            if conf < MIN_CONFIDENCE:
                continue
            x1, y1, x2, y2 = [int(x) for x in box.xyxy[0]]
            if ((x2 - x1) * (y2 - y1)) / img_area < MIN_AREA_RATIO:
                continue
            if conf > max_conf:
                max_conf   = conf
                cls_id     = int(box.cls[0])
                label_name = res.names.get(cls_id, "unknown")
                best_det   = {
                    "confidence": conf,
                    "bbox":       [x1, y1, x2, y2],
                    "label":      label_name,
                }
                #self._write_yolo_log(video_path, cam_id, label_name, conf, [x1, y1, x2, y2])

        if not best_det:
            return None

        # Salva crop
        frames_dir.mkdir(parents=True, exist_ok=True)
        crop_path = frames_dir / f"{image_path.stem}_crop{image_path.suffix}"
        try:
            img = cv2.imread(str(image_path))
            if img is not None:
                cx1, cy1, cx2, cy2 = get_crop_coordinates(best_det["bbox"], img.shape)
                cv2.imwrite(str(crop_path), img[cy1:cy2, cx1:cx2])
                best_det["crop_path"] = str(crop_path)
            else:
                best_det["crop_path"] = None
        except Exception as e:
            log.critical(f"Error cropping NVR image: {e}")
            best_det["crop_path"] = None

        ts = datetime.fromtimestamp(image_path.stat().st_mtime)

        return {
            "camera_id":        cam_id,
            "video_path":       str(video_path),
            "categories_found": ["person"],
            "frames": [{
                "category":   "person",
                "frame_path": str(image_path),
                "crop_path":  best_det.get("crop_path"),
                "confidence": best_det["confidence"],
                "bbox":       best_det["bbox"],
                "timestamp":  ts.isoformat(),
            }],
            "resolved_by": "nvr_image",
        }

    # ------------------------------------------------------------------
    # Scan video principale
    # ------------------------------------------------------------------

    def scan_video(self, video_path: Path, frames_dir: Path, cam_id: str) -> Optional[dict]:
        """
        Analizza un video con stride adattivo e early exit.

        Novità rispetto alla versione precedente:
          - Soglie giorno/notte per telecamera (thresholds / thresholds_night)
          - Stride calcolato in secondi reali (vid_stride_sec)
          - Warmup iniziale a stride ridotto (primi warmup_sec secondi)
          - Dynamic stride invariato (opzionale per telecamera)
        """
        # ------------------------------------------------------------------
        # 1. Configurazione e Target IDs
        # ------------------------------------------------------------------
        cam_cfg       = self.cameras_config.get(str(cam_id), {})
        ignore_labels = cam_cfg.get("filters", {}).get("ignore_labels", [])

        self.ensure_model_loaded()

        final_target_ids = get_target_ids(
            model=self.model,
            settings=self.settings,
            mode=self.mode,
            camera_ignore_labels=ignore_labels,
        )
        if not final_target_ids:
            log.warning(f"Nessuna classe target per cam={cam_id}")
            return None

        # ------------------------------------------------------------------
        # 2. Apertura Video e Parametri Temporali
        # ------------------------------------------------------------------
        cap = get_video_capture(video_path)
        if cap is None:
            return None

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        # Soglie giorno/notte basate sul mtime del video
        video_ts   = datetime.fromtimestamp(video_path.stat().st_mtime)
        thresholds = self._get_thresholds(cam_cfg, video_ts)

        # Stride in secondi reali → frame
        # vid_stride_sec ha la precedenza su vid_stride (retrocompatibilità)
        stride_sec = self.yolo_cfg.get("vid_stride_sec", None)
        if stride_sec is not None:
            stride_std = max(1, round(fps * stride_sec))
        else:
            stride_std = self.yolo_cfg.get("vid_stride", 12)

        time_gap_sec            = self.yolo_cfg.get("time_gap_sec", 3)
        occ_gap_frames          = int(fps * time_gap_sec)
        num_occurrence          = self.yolo_cfg.get("num_occurrence", 3)
        yolo_override_threshold = self.yolo_cfg.get("yolo_override_threshold", 0.85)

        # ------------------------------------------------------------------
        # 3. Strutture Dati
        # ------------------------------------------------------------------
        detection_groups = self.yolo_cfg.get("detection_groups", {})

        label_to_group = {
            lbl: group_name.lower()
            for group_name, labels in detection_groups.items()
            for lbl in labels
        }

        detections       = {g.lower(): [] for g in detection_groups}
        occ_count        = {g.lower(): 0  for g in detection_groups}
        last_saved_frame = {g.lower(): -occ_gap_frames for g in detection_groups}

        frame_idx             = 0
        self.last_detection_idx = -9999
        prev_stride           = stride_std

        # ------------------------------------------------------------------
        # 4. Loop di Analisi
        # ------------------------------------------------------------------
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            found_valid_target = False

            yolo_results = self.model.predict(
                source=frame,
                classes=final_target_ids,
                verbose=False,
                conf=0.20,
                device=self.device,
            )[0]

            if len(yolo_results.boxes) > 0:
                for box in yolo_results.boxes:
                    cls_id = int(box.cls[0])
                    conf   = float(box.conf[0])
                    label  = self.model.names[cls_id]
                    group  = label_to_group.get(label)

                    if not group or conf < thresholds.get(group, 0.50):
                        continue

                    found_valid_target      = True
                    self.last_detection_idx = frame_idx

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    bbox_area  = (x2 - x1) * (y2 - y1)
                    h, w_frame = frame.shape[:2]
                    area_ratio = bbox_area / (h * w_frame)

                    detections[group].append({
                        "frame_idx":     frame_idx,
                        "confidence":    conf,
                        "bbox":          [x1, y1, x2, y2],
                        "area":          bbox_area,
                        "area_ratio":    area_ratio,
                        "yolo_label":    label,
                        "yolo_reliable": conf >= yolo_override_threshold,
                    })

                    if group in ("person", "animal"):
                        if frame_idx - last_saved_frame[group] >= occ_gap_frames:
                            last_saved_frame[group] = frame_idx
                            occ_count[group] += 1

            # Early exit
            if occ_count["person"] >= num_occurrence or occ_count["animal"] >= num_occurrence:
                log.debug(f"Early stop su video={video_path.name}")
                break

            # Calcolo stride per il prossimo salto
            current_stride = self._get_next_stride(
                frame_idx, fps, found_valid_target, cam_cfg, stride_std
            )
            if current_stride != prev_stride:
                prev_stride = current_stride
            #print(f"cambio stride: {current_stride} frame {frame_idx}")
            # Skip frame (grab è veloce, non decodifica)
            if current_stride > 1:
                for _ in range(current_stride - 1):
                    if not cap.grab():
                        break
                    frame_idx += 1

            frame_idx += 1

        cap.release()

        # ------------------------------------------------------------------
        # 5. Estrazione Frame Finali
        # ------------------------------------------------------------------
        if not any(detections.values()):
            return {
                "camera_id":        cam_id,
                "video_path":       str(video_path),
                "categories_found": ["others"],
                "frames":           [],
                "timestamp":        datetime.now().isoformat(),
            }

        frames_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        saved_frames = extract_frames_with_cache(
            cap=cap,
            detections=detections,
            fps=fps,
            video_path=video_path,
            frames_dir=frames_dir,
            frames_per_category=num_occurrence,
        )
        cap.release()

        if not saved_frames:
            return {
                "camera_id":        cam_id,
                "video_path":       str(video_path),
                "categories_found": ["others"],
                "frames":           [],
                "timestamp":        datetime.now().isoformat(),
            }

        categories_found = [cat for cat, found_list in detections.items() if found_list]

        return {
            "camera_id":        cam_id,
            "video_path":       str(video_path),
            "categories_found": categories_found,
            "frames":           saved_frames,
            "timestamp":        datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------
    # Stride adattivo
    # ------------------------------------------------------------------

    def _get_next_stride(self, frame_idx: int, fps: float,
                         found_valid_target: bool, cam_config: dict,
                         stride_std: int) -> int:
        """
        Calcola lo stride per il prossimo salto di frame.

        Logica in ordine di priorità:
          1. Warmup: primi warmup_sec secondi → stride ridotto (alta sensibilità)
          2. Telecamera senza dynamic_stride → stride standard
          3. Detection attiva → stride standard
          4. Pre-roll (zona protetta dopo warmup) → stride standard
          5. Cooldown dopo ultima detection → stride standard
          6. Velocità crociera → stride_fast
        """
        dyn_settings = self.yolo_cfg.get("dynamic_stride_settings", {})

        # 1. Warmup iniziale: sempre attivo, indipendentemente dal dynamic_stride
        warmup_sec = dyn_settings.get("warmup_sec", 5)
        if frame_idx < int(warmup_sec * fps):
            return max(1, round(fps * 0.25))  # ~4 frame/sec nei primi N secondi

        # 2. Senza dynamic_stride: stride standard per tutta la durata
        if not cam_config.get("dynamic_stride", False):
            return stride_std

        # 3. Parametri dynamic stride
        # Dopo
        stride_fast_sec = dyn_settings.get("stride_fast_sec", 1.0)
        stride_fast = max(1, round(fps * stride_fast_sec))
        pre_roll_sec = dyn_settings.get("pre_roll_sec", 20)
        cooldown_sec = dyn_settings.get("cooldown_sec", 5)

        # 4. Detection attiva → attenzione massima
        if found_valid_target:
            return stride_std

        # 5. Pre-roll (zona protetta dopo il warmup)
        if frame_idx < int(pre_roll_sec * fps):
            return stride_std

        # 6. Cooldown dopo l'ultima detection
        if (frame_idx - self.last_detection_idx) < int(cooldown_sec * fps):
            return stride_std

        # 7. Velocità crociera: niente trovato da un po'
        return stride_fast

    # ------------------------------------------------------------------
    # Scan fallback (bassa confidenza su immagine NVR)
    # ------------------------------------------------------------------

    def scan_fallback(self, image_path: Path, target_ids: list) -> Optional[dict]:
        """
        Scan ad alta sensibilità su immagine NVR per il fallback.
        Usato per recuperare eventi che YOLO ha mancato al primo passaggio.
        """
        res = self.model(
            str(image_path), classes=target_ids,
            conf=0.15, verbose=False, device=self.device
        )[0]

        if len(res.boxes) == 0:
            log.debug(f"YOLO Fallback: niente trovato su img={image_path.name}")
            return None

        best_box = res.boxes[0]
        return {
            "confidence":    float(best_box.conf[0]),
            "bbox":          [int(x) for x in best_box.xyxy[0]],
            "yolo_reliable": False,
        }

    # ------------------------------------------------------------------
    # Scan a bassissima confidenza (per run_fallback_phase)
    # ------------------------------------------------------------------

    def low_conf_image_scan(self, image_path: Path, video_path: Path,
                            cam_id: str) -> Optional[dict]:
        """
        Scan a confidenza molto bassa (0.05) su immagine NVR.
        Usato nella fase di fallback avanzato per recuperare eventi borderline.
        """
        cam_cfg       = self.cameras_config.get(str(cam_id), {})
        ignore_labels = cam_cfg.get("filters", {}).get("ignore_labels", [])

        target_ids = get_target_ids(
            model=self.model,
            settings=self.settings,
            mode=self.mode,
            camera_ignore_labels=ignore_labels,
        )

        results = self.model(
            str(image_path), classes=target_ids,
            conf=0.05, verbose=False, device=self.device
        )

        if not results or len(results[0].boxes) == 0:
            log.debug(f"YOLO low_conf: niente trovato su img={image_path.name}")
            return None

        best_det   = None
        max_conf   = -1.0
        label_name = "unknown"

        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf > max_conf:
                max_conf   = conf
                cls_id     = int(box.cls[0])
                label_name = self.model.names.get(cls_id, "unknown")
                best_det   = {
                    "category":   label_name.lower(),
                    "confidence": conf,
                    "bbox":       [int(x) for x in box.xyxy[0]],
                }

        if not best_det:
            return None

        #self._write_yolo_log(video_path, cam_id, label_name, max_conf, best_det["bbox"])

        return {
            "camera_id":     cam_id,
            "video_name":    Path(video_path).name,
            "video_path":    video_path,
            "image_path":    image_path,
            "yolo_category": best_det["category"],
            "yolo_data":     best_det,
        }

    # # ------------------------------------------------------------------
    # # Fase fallback completa (YOLO + Vision su video non classificati)
    # # ------------------------------------------------------------------

    # def run_fallback_phase(self, current_results: list) -> list:
    #     """
    #     Fase di fallback: rilancia YOLO a bassa confidenza sui video rimasti
    #     in OTHERS, poi passa i sospetti alla Vision AI per conferma.
    #     """
    #     log.info("Start FALLBACK phase for videos not recognized in prev steps.")

    #     fallback_targets = self._get_fallback_targets(current_results)
    #     if not fallback_targets:
    #         return current_results

    #     for item in fallback_targets:
    #         suspicious_frames = []
    #         target_ids = self.yolo_engine.get_target_ids(item["camera_id"])

    #         for img_path in item["nvr_images"]:
    #             detection = self.scan_fallback(img_path, target_ids)
    #             if detection:
    #                 suspicious_frames.append({
    #                     "video_name": item["video_name"],
    #                     "video_path": item["video_path"],
    #                     "camera_id":  item["camera_id"],
    #                     "frame_path": img_path,
    #                     "yolo_data":  detection,
    #                 })

    #         for suspect in suspicious_frames:
    #             log.debug(f"Fallback: Vision AI check suspect in video={suspect['video_name']}")
    #             report = self.vision_engine.refine_single_video(suspect)
    #             if report and report.get("category") != "nothing":
    #                 log.debug(f"Found video={item['video_name']} category={report['category']}")
    #                 current_results.append(report)
    #                 break

    #     return current_results