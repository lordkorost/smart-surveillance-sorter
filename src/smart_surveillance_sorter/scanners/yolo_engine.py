"""
YoloEngine — YOLO inference engine for smart surveillance
============================================================

Main features:
  - Lazy loading of model (loads only when needed)
  - scan_single_image: analyze NVR images with high confidence (>=0.60)
  - scan_video: analyze videos with adaptive stride and early exit
  - Day/night thresholds per camera (thresholds / thresholds_night)
  - Stride based on actual video FPS (vid_stride_sec in settings)
  - Initial warmup at reduced stride (first warmup_sec seconds)
  - Dynamic stride for quiet cameras (e.g. parking lot)
  - low_conf_image_scan: low confidence scan for fallback
"""

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
    """YOLO detection engine with lazy loading, adaptive stride, and day/night thresholds."""

    def __init__(self, mode, device, settings, cameras_config):
        """Initialize YOLO engine.
        
        Args:
            mode: Detection mode (full, person, person_animal)
            device: Computing device (cuda or cpu)
            settings: Application settings dictionary
            cameras_config: Camera configuration dictionary
        """
        self.mode           = mode
        self.device         = device
        self.settings       = settings
        self.cameras_config = cameras_config

        # Lazy loading — model is loaded only on first call
        self.model      = None
        self.name_to_id = None
        self.yolo_cfg   = self.settings.get("yolo_settings", {})

        # Coordinates for astronomical night calculation (used for day/night thresholds)
        city = self.settings.get("city", "Roma")
        self.lat, self.lon = get_smart_coordinates(city)

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def ensure_model_loaded(self):
        """Load model to GPU/CPU only if not already present (lazy loading).
        
        Loads the YOLO model into VRAM only on first call to optimize memory usage.
        """
        if self.model is None:
            self.model = load_smart_yolo(
                model_name=self.yolo_cfg.get("model_path", "yolov8l"),
                device=self.device,
            )
            self.name_to_id = {v: k for k, v in self.model.names.items()}

    # ------------------------------------------------------------------
    # Helper: day/night thresholds per camera
    # ------------------------------------------------------------------

    def _get_thresholds(self, cam_cfg: dict, timestamp: datetime) -> dict:
        """Get correct thresholds based on current time (day vs night).
        
        If camera doesn't have 'thresholds_night', uses day thresholds instead.
        Determines day/night based on astronomical sunset/sunrise.
        
        Args:
            cam_cfg: Camera configuration dict
            timestamp: Frame timestamp with timezone info
            
        Returns:
            Threshold dictionary (day or night version)
        """
        thresholds_day   = cam_cfg.get("thresholds", {})
        thresholds_night = cam_cfg.get("thresholds_night", thresholds_day)
        is_night = is_night_astronomic(timestamp, self.lat, self.lon)
        return thresholds_night if is_night else thresholds_day

    # ------------------------------------------------------------------
    # Scan NVR image (high confidence)
    # ------------------------------------------------------------------

    def scan_single_image(self, image_path: Path, video_path: Path,
                          frames_dir: Path, cam_id: str) -> Optional[dict]:
        """Analyze a single NVR image with high confidence threshold (>=0.60).
        
        Used for fast-track: if NVR image is already sufficient,
        the video is classified without opening it.
        
        Args:
            image_path: Path to NVR image
            video_path: Associated video file path
            frames_dir: Directory for saving extracted frames
            cam_id: Camera identifier
            
        Returns:
            Detection result dict or None if no promising detections
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
                

        if not best_det:
            return None

        # Save crop
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
        Analyze video with adaptive stride and early exit.

        Changes from previous version:
          - Day/night thresholds per camera (thresholds / thresholds_night)
          - Stride calculated in real seconds (vid_stride_sec)
          - Initial warmup with reduced stride (first warmup_sec seconds)
          - Dynamic stride unchanged (optional per camera)
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
            log.warning(f"No target class for camera={cam_id}")
            return None

        # ------------------------------------------------------------------
        # 2. Apertura Video e Parametri Temporali
        # ------------------------------------------------------------------
        cap = get_video_capture(video_path)
        if cap is None:
            return None

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        # Day/night thresholds based on video mtime
        video_ts   = datetime.fromtimestamp(video_path.stat().st_mtime)
        thresholds = self._get_thresholds(cam_cfg, video_ts)

        # Stride in real seconds → frame
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
        # 4. Analysis Loop
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

            # Calculate stride for next jump
            current_stride = self._get_next_stride(
                frame_idx, fps, found_valid_target, cam_cfg, stride_std
            )
            if current_stride != prev_stride:
                prev_stride = current_stride
            
            if current_stride > 1:
                for _ in range(current_stride - 1):
                    if not cap.grab():
                        break
                    frame_idx += 1

            frame_idx += 1

        cap.release()

        # ------------------------------------------------------------------
        # 5. Final frame extraction
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
    # Adaptive stride
    # ------------------------------------------------------------------

    def _get_next_stride(self, frame_idx: int, fps: float,
                         found_valid_target: bool, cam_config: dict,
                         stride_std: int) -> int:
        """
        Calculate stride for next frame jump.

        Priority logic:
          1. Warmup: first warmup_sec seconds → reduced stride (high sensitivity)
          2. Camera without dynamic_stride → standard stride
          3. Detection active → standard stride
          4. Pre-roll (protected zone after warmup) → standard stride
          5. Cooldown after last detection → standard stride
          6. Cruising speed → stride_fast
        """
        dyn_settings = self.yolo_cfg.get("dynamic_stride_settings", {})

        # 1. Initial warmup: always active, regardless of dynamic_stride
        warmup_sec = dyn_settings.get("warmup_sec", 5)
        if frame_idx < int(warmup_sec * fps):
            return max(1, round(fps * 0.25))  # ~4 frame/sec in first N seconds

        # 2. Without dynamic_stride: standard stride for entire duration
        if not cam_config.get("dynamic_stride", False):
            return stride_std

        # 3. Parametri dynamic stride
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

        # 6. Cooldown after last detection
        if (frame_idx - self.last_detection_idx) < int(cooldown_sec * fps):
            return stride_std

        # 7. Cruising speed: nothing found for a while
        return stride_fast

    # ------------------------------------------------------------------
    # Scan fallback (low confidence on NVR image)
    # ------------------------------------------------------------------

    def scan_fallback(self, image_path: Path, target_ids: list) -> Optional[dict]:
        """
        High sensitivity scan on NVR image for fallback recovery.
        Used to recover events that YOLO missed in the first pass.
        """
        res = self.model(
            str(image_path), classes=target_ids,
            conf=0.15, verbose=False, device=self.device
        )[0]

        if len(res.boxes) == 0:
            log.debug(f"YOLO Fallback: nothing found su img={image_path.name}")
            return None

        best_box = res.boxes[0]
        return {
            "confidence":    float(best_box.conf[0]),
            "bbox":          [int(x) for x in best_box.xyxy[0]],
            "yolo_reliable": False,
        }

    # ------------------------------------------------------------------
    # Very low confidence scan (for fallback phase)
    # ------------------------------------------------------------------

    def low_conf_image_scan(self, image_path: Path, video_path: Path,
                            cam_id: str) -> Optional[dict]:
        """
        Very low confidence scan (0.05) on NVR image.
        Used in advanced fallback phase to recover borderline events.
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
            log.debug(f"YOLO low_conf: nothing found su img={image_path.name}")
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


        return {
            "camera_id":     cam_id,
            "video_name":    Path(video_path).name,
            "video_path":    video_path,
            "image_path":    image_path,
            "yolo_category": best_det["category"],
            "yolo_data":     best_det,
        }
