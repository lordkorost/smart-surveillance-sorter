"""
ClipBlipEngine v2 — Sistema di scoring semplificato
====================================================

Principi:
  - PERSON ha priorità assoluta (soglia bassa, fake penalty leggera)
  - Un solo flusso decisionale per frame (no if annidati multipli)
  - Tutti i parametri in clip_blip_settings_v2.json, documentati
  - Fix falsi negativi: bbox_size_bonus per oggetti piccoli (teste angolate)

Rispetto alla v1:
  - Eliminati: PERSON_FAKE_RELATIVE, PERSON_BOOST_TOLERANCE, DELTA_THRESHOLD,
               PERSON_PRIORITY_THRESHOLD, DAY_ANIMAL_MIN_CONF, DAY_ANIMAL_MARGIN,
               ANIMAL_AGG_THRESHOLDS, VEHICLE_AGG_THRESHOLDS
  - Sostituiti da: FAKE_PENALTY_WEIGHT per classe + THRESHOLD per classe
  - _calculate_day_category e _calculate_night_category → unico _decide_frame_label
"""

import logging
import time
import torch
from datetime import datetime
from transformers import BlipConfig, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from open_clip import create_model_and_transforms, tokenize
from smart_surveillance_sorter.constants import CLIP_BLIP_JSON
from smart_surveillance_sorter.utils import get_smart_coordinates, is_night_astronomic, load_json

log = logging.getLogger(__name__)


class ClipBlipEngine:
    def __init__(self, settings, cameras_config, mode, device):
        self.clip_blip_settings = load_json(CLIP_BLIP_JSON)
        self.DEVICE = device
        self.mode = mode
        self.settings = settings
        self.ch_configs = cameras_config

        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

        # --- CLIP ---
        clip_m_name = self.clip_blip_settings.get("clip_model", "ViT-L-14")
        clip_pre = self.clip_blip_settings.get("clip_pretrained", "openai")
        self.clip_model, self.preprocess, _ = create_model_and_transforms(clip_m_name, pretrained=clip_pre)
        self.clip_model.to(self.DEVICE).eval()

        # --- BLIP ---
        blip_m_name = self.clip_blip_settings.get("blip_model", "Salesforce/blip-image-captioning-base")
        config = BlipConfig.from_pretrained(blip_m_name)
        config.tie_word_embeddings = False
        self.blip_processor = BlipProcessor.from_pretrained(blip_m_name)
        self.blip_model = BlipForConditionalGeneration.from_pretrained(blip_m_name, config=config).to(self.DEVICE).eval()

        cb = self.clip_blip_settings

        # Classi principali e dizionari di supporto
        self.MAIN_CLASSES  = cb.get("MAIN_CLASSES", ["PERSON", "ANIMAL", "VEHICLE"])
        self.FAKE_KEYS     = cb.get("FAKE_KEYS", {})
        self.BLIP_KEYWORDS = cb.get("BLIP_KEYWORDS", {})


        # Soglia minima di score CLIP perché un frame sia "significativo"
        # (evita che frame completamente vuoti guidino la decisione)
        self.SIGNIFICANCE_THRESHOLD = cb.get("SIGNIFICANCE_THRESHOLD", 0.15)

        # Pesi crop vs frame intero nel calcolo dello score CLIP
        self.FINAL_WEIGHT_CROP  = cb.get("FINAL_WEIGHT_CROP", 0.7)
        self.FINAL_WEIGHT_FRAME = cb.get("FINAL_WEIGHT_FRAME", 0.3)

        # Boost BLIP quando la caption contiene keyword di quella classe
        self.BLIP_BOOST = cb.get("BLIP_BOOST", {"PERSON": 0.35, "ANIMAL": 0.10, "VEHICLE": 0.10})

        # Soglia di score finale (dopo boost e penalty) per classificare un frame
        # PERSON: valore basso = preferiamo avere falsi positivi a falsi negativi
        self.THRESHOLD = cb.get("THRESHOLD", {"PERSON": 0.15, "ANIMAL": 0.35, "VEHICLE": 0.30})

        # Quanto i fake score penalizzano lo score finale di ogni classe
        # Valore basso = classe favorita (fake non la penalizzano molto)
        # Valore alto  = classe penalizzata dai fake
        self.FAKE_PENALTY_WEIGHT = cb.get("FAKE_PENALTY_WEIGHT", {"PERSON": 0.3, "ANIMAL": 0.5, "VEHICLE": 0.6})

        # Fix falsi negativi: bonus persona se bbox è piccolo (testa nell'angolo)
        # BBOX_SMALL_RATIO: se bbox_area / frame_area < questa soglia → oggetto piccolo
        # BBOX_SMALL_PERSON_BONUS: bonus aggiunto allo score PERSON
        self.BBOX_SMALL_RATIO        = cb.get("BBOX_SMALL_RATIO", 0.04)   # < 4% del frame
        self.BBOX_SMALL_PERSON_BONUS = cb.get("BBOX_SMALL_PERSON_BONUS", 0.15)

        # Boost notturno per PERSON (la notte è più difficile per CLIP)
        self.YOLO_NIGHT_BOOST = cb.get("YOLO_NIGHT_BOOST", 0.30)

        # Soglie aggregazione video: quanti frame servono e con che score medio
        # per classificare l'intero video come ANIMAL o VEHICLE
        self.ANIMAL_START_THRESHOLD   = cb.get("ANIMAL_START_THRESHOLD", 0.45)
        self.ANIMAL_STEP_REDUCTION    = cb.get("ANIMAL_STEP_REDUCTION", 0.05)
        self.ANIMAL_MIN_THRESHOLD     = cb.get("ANIMAL_MIN_THRESHOLD", 0.15)
        self.VEHICLE_START_THRESHOLD  = cb.get("VEHICLE_START_THRESHOLD", 0.50)
        self.VEHICLE_STEP_REDUCTION   = cb.get("VEHICLE_STEP_REDUCTION", 0.05)
        self.VEHICLE_MIN_THRESHOLD    = cb.get("VEHICLE_MIN_THRESHOLD", 0.20)

        # Coordinate per calcolo notte astronomica
        self.city_name = self.settings.get("city", "Roma")
        self.lat, self.lon = get_smart_coordinates(self.city_name)
        log.debug(f"ClipBlipEngine  — city={self.city_name} ({self.lat}, {self.lon})")

        # Mappa label YOLO → classe principale
        self.label_to_main_class = {}
        groups = self.settings.get("yolo_settings", {}).get("detection_groups", {})
        for main_cls, labels in groups.items():
            for label in labels:
                self.label_to_main_class[label.lower()] = main_cls

        self.results = {}

    # ------------------------------------------------------------------
    # CLIP helpers
    # ------------------------------------------------------------------

    def _get_clip_score(self, image_tensor, texts):
        with torch.no_grad():
            img_feat  = self.clip_model.encode_image(image_tensor)
            txt_feat  = self.clip_model.encode_text(tokenize(texts).to(self.DEVICE))
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
            sim = (100.0 * img_feat @ txt_feat.T).softmax(dim=-1)
            return {cls: float(sim[0, i]) for i, cls in enumerate(texts)}

    # ------------------------------------------------------------------
    # Scoring per singolo frame 
    # ------------------------------------------------------------------

    def _score_frame(self, frame, active_rules, is_night):
        """
        Calcola lo score finale per un frame e restituisce la label.

        Flusso:
          1. CLIP score (crop pesato + frame intero)
          2. BLIP boost (keyword nella caption)
          3. Fake penalty (sottrai fake_score * peso)
          4. Bbox small bonus (fix teste piccole)
          5. Night boost (PERSON di notte)
          6. Confronta con soglia → label
        """
        yolo_category = frame.get("category", "").upper()
        current_main_class = [yolo_category] if yolo_category in self.MAIN_CLASSES else self.MAIN_CLASSES

        frame_path = frame.get("frame_path")
        crop_path  = frame.get("crop_path")
        bbox       = frame.get("bbox")  # [x1, y1, x2, y2] in pixel assoluti

        # --- Immagini ---
      
        try:
            crop_img  = self.preprocess(Image.open(crop_path).convert("RGB")).unsqueeze(0).to(self.DEVICE)
            #log.debug(f"Loading frame: {frame_path}")
            frame_img = self.preprocess(Image.open(frame_path).convert("RGB")).unsqueeze(0).to(self.DEVICE) if frame_path else crop_img

            # --- BLIP caption ---
            raw_img = Image.open(crop_path).convert("RGB")
            blip_inputs = self.blip_processor(images=raw_img, return_tensors="pt").to(self.DEVICE)
            caption = self.blip_processor.decode(self.blip_model.generate(**blip_inputs)[0], skip_special_tokens=True)
        except Exception as e:
            log.warning(f"Skipping missing/corrupt frame: {crop_path} — {e}")
            return None
        
        
       
        
        # --- CLIP scores su crop e frame ---
        fake_prompts = [desc for descs in self.FAKE_KEYS.values() for desc in descs]
        all_prompts  = current_main_class + fake_prompts
        #t0 = time.time()
        clip_crop  = self._get_clip_score(crop_img, all_prompts)
        
        clip_frame = self._get_clip_score(frame_img, all_prompts)
        

        # Pesi crop/frame (sovrascrivibili per camera)
        w_crop  = active_rules["FINAL_WEIGHT_CROP"]
        w_frame = active_rules["FINAL_WEIGHT_FRAME"]

        # --- Fake scores (pesati per camera se specificato) ---
        fake_weights = active_rules.get("FAKE_WEIGHTS", {})
        fake_scores = {
            fk: max(clip_crop[d] for d in descs) * fake_weights.get(fk, 1.0)
            for fk, descs in self.FAKE_KEYS.items()
        }
        max_fake_score = max(fake_scores.values()) if fake_scores else 0.0

        # --- Score finale per ogni classe ---
        final_scores = {cls: 0.0 for cls in self.MAIN_CLASSES}

        for cls in current_main_class:
            # 1. Base: media pesata crop + frame
            base = w_crop * clip_crop.get(cls, 0.0) + w_frame * clip_frame.get(cls, 0.0)

            # 2. BLIP boost se caption contiene keyword
            blip_boost = active_rules["BLIP_BOOST"].get(cls, 0.0)
            has_keyword = any(k.lower() in caption.lower() for k in self.BLIP_KEYWORDS.get(cls, []))
            blip_bonus = blip_boost if has_keyword else 0.0

            # 3. Fake penalty (più alta per classi non prioritarie)
            penalty_weight = active_rules["FAKE_PENALTY_WEIGHT"].get(cls, 0.7)
            fake_penalty = max_fake_score * penalty_weight

            final_scores[cls] = max(0.0, base + blip_bonus - fake_penalty)

        if bbox and yolo_category == "PERSON" and final_scores["PERSON"] > 0.05:
            bbox_bonus = self._get_bbox_small_bonus(bbox, frame_path, active_rules)
            final_scores["PERSON"] += bbox_bonus

        # 5. Night boost per PERSON
        if is_night and yolo_category == "PERSON":
            final_scores["PERSON"] += active_rules["YOLO_NIGHT_BOOST"]

        # --- Decisione label ---
        label = self._decide_frame_label(final_scores, yolo_category, active_rules)

        return {
            "clip_crop":      clip_crop,
            "clip_frame":     clip_frame,
            "blip_caption":   caption,
            "fake_scores":    fake_scores,
            "max_fake_score": max_fake_score,
            "final_scores":   final_scores,
            "label":          label,
            "bbox":           bbox,
        }

    def _get_bbox_small_bonus(self, bbox, frame_path, active_rules):
        """
        Se il bbox della persona occupa meno di BBOX_SMALL_RATIO del frame,
        aggiunge un bonus allo score PERSON.
        Questo corregge i falsi negativi dove si vede solo la testa in un angolo.
        """
        try:
            with Image.open(frame_path) as img:
                fw, fh = img.size
            x1, y1, x2, y2 = bbox
            bbox_area  = max(0, x2 - x1) * max(0, y2 - y1)
            frame_area = fw * fh
            if frame_area == 0:
                return 0.0
            ratio = bbox_area / frame_area
            threshold = active_rules.get("BBOX_SMALL_RATIO", self.BBOX_SMALL_RATIO)
            bonus     = active_rules.get("BBOX_SMALL_PERSON_BONUS", self.BBOX_SMALL_PERSON_BONUS)
            return bonus if ratio < threshold else 0.0
        except Exception:
            return 0.0

    def _decide_frame_label(self, final_scores, yolo_category, active_rules):
        """
        Regola semplice:
          - Scorre le classi in ordine di priorità (PERSON > VEHICLE > ANIMAL)
          - La prima che supera la sua soglia vince
          - PERSON ha soglia bassa → priorità massima
        """
        thresholds = active_rules["THRESHOLD"]
        priority   = ["PERSON", "ANIMAL", "VEHICLE"]

        for cls in priority:
            if cls not in final_scores:
                continue
            # Considera solo la classe rilevata da YOLO (o tutte se YOLO non ha match)
            if yolo_category in self.MAIN_CLASSES and cls != yolo_category:
                continue
            if final_scores[cls] >= thresholds.get(cls, 0.3):
                return cls

        return "OTHERS"

    # ------------------------------------------------------------------
    # Scan singolo video
    # ------------------------------------------------------------------

    def scan_single_video(self, video_data):
        frames = video_data.get("frames", [])
        if not frames:
            log.debug(f"  (Nessun frame per video={video_data.get('video_path')})")
            return {}

        # Fast-track NVR: se il NVR stesso ha già classificato come persona, fiducia piena
        is_nvr = video_data.get("resolved_by") == "nvr_image"
        has_yolo_person = any(f.get("category") == "person" for f in frames)
        if is_nvr and has_yolo_person:
            best_f = next(f for f in frames if f.get("category") == "person")
            frame_res = {
                "clip_crop":    {"PERSON": 1.0, "ANIMAL": 0.0, "VEHICLE": 0.0},
                "clip_frame":   {"PERSON": 1.0, "ANIMAL": 0.0, "VEHICLE": 0.0},
                "blip_caption": f"yolo_nvr_validated_conf_{best_f.get('confidence', 0):.2f}",
                "fake_scores":  {},
                "max_fake_score": 0.0,
                "final_scores": {"PERSON": 1.0, "ANIMAL": 0.0, "VEHICLE": 0.0},
                "label":        "PERSON",
                "bbox":         best_f.get("bbox"),
            }
            video_path = video_data.get("video_path")
            return {video_path: {"frames": [frame_res], "video_category": "PERSON"}}

        # Regole attive (default + override per camera)
        camera_id    = video_data.get("camera_id")
        cam_config   = self.ch_configs.get(camera_id, {})
        
        active_rules = self._get_active_rules(camera_id)
      

        ignore_classes = set(cam_config.get("filters", {}).get("ignore_classes", []))

        frames_list = []
        for frame in frames:
            yolo_category = frame.get("category", "").upper()
            
            if yolo_category in ignore_classes:
                continue

            dt       = datetime.fromisoformat(frame.get("timestamp"))
            is_night = is_night_astronomic(dt, self.lat, self.lon)

            frame_res = self._score_frame(frame, active_rules, is_night)
            if frame_res is None:
                continue
            frames_list.append(frame_res)

        video_path = video_data.get("video_path")
        video_dict = {video_path: {"frames": frames_list}}
        video_dict[video_path]["video_category"] = self._decide_video_category(frames_list, active_rules)

        return video_dict

    # ------------------------------------------------------------------
    # Aggregazione video
    # ------------------------------------------------------------------

    def _decide_video_category(self, frames_list, active_rules):
        """
        Priorità: PERSON > VEHICLE > ANIMAL
        PERSON: basta UN frame con label PERSON
        ANIMAL/VEHICLE: score medio > soglia dinamica (diminuisce con più frame)
        """
        # 1. Persona: basta un frame
        if any(f["label"] == "PERSON" for f in frames_list):
            return "PERSON"

        # 2. Animale
        animal_scores = [f["final_scores"]["ANIMAL"] for f in frames_list if f["label"] == "ANIMAL"]
        if animal_scores:
            threshold = self._get_dynamic_threshold(len(animal_scores), "ANIMAL", active_rules)
            if (sum(animal_scores) / len(animal_scores)) > threshold:
                return "ANIMAL"
            
         # 3. Veicolo 
        vehicle_scores = [f["final_scores"]["VEHICLE"] for f in frames_list if f["label"] == "VEHICLE"]
        if vehicle_scores:
            threshold = self._get_dynamic_threshold(len(vehicle_scores), "VEHICLE", active_rules)
            if (sum(vehicle_scores) / len(vehicle_scores)) > threshold:
                return "VEHICLE"

        return "OTHERS"

    def _get_dynamic_threshold(self, count, category, rules):
        """Soglia che scende all'aumentare dei frame: più evidenza → soglia più bassa."""
        prefix  = category.upper()
        start   = rules.get(f"{prefix}_START_THRESHOLD", 0.45)
        step    = rules.get(f"{prefix}_STEP_REDUCTION", 0.05)
        min_t   = rules.get(f"{prefix}_MIN_THRESHOLD", 0.15)
        return max(min_t, start - (max(1, count) - 1) * step)

    # ------------------------------------------------------------------
    # Regole attive (default + override per camera)
    # ------------------------------------------------------------------

    def _get_active_rules(self, camera_id):
        """
        Ritorna le regole attive per la camera specificata.
        Ogni valore nel JSON della camera sovrascrive il default globale.
        """
        custom = self.ch_configs.get(camera_id, {}).get("blip_rules", {})

        def c(key, default):
            return custom.get(key, default)

        return {
            "SIGNIFICANCE_THRESHOLD": c("SIGNIFICANCE_THRESHOLD", self.SIGNIFICANCE_THRESHOLD),
            "FINAL_WEIGHT_CROP":      c("FINAL_WEIGHT_CROP",      self.FINAL_WEIGHT_CROP),
            "FINAL_WEIGHT_FRAME":     c("FINAL_WEIGHT_FRAME",     self.FINAL_WEIGHT_FRAME),

            # Dict — merge: default + override camera
            "BLIP_BOOST":         {**self.BLIP_BOOST,         **c("BLIP_BOOST", {})},
            "THRESHOLD":          {**self.THRESHOLD,          **c("THRESHOLD", {})},
            "FAKE_PENALTY_WEIGHT":{**self.FAKE_PENALTY_WEIGHT,**c("FAKE_PENALTY_WEIGHT", {})},

            "YOLO_NIGHT_BOOST": c("YOLO_NIGHT_BOOST", self.YOLO_NIGHT_BOOST),

            # Fix teste piccole
            "BBOX_SMALL_RATIO":        c("BBOX_SMALL_RATIO",        self.BBOX_SMALL_RATIO),
            "BBOX_SMALL_PERSON_BONUS": c("BBOX_SMALL_PERSON_BONUS", self.BBOX_SMALL_PERSON_BONUS),

            # Aggregazione video
            "ANIMAL_START_THRESHOLD":  c("ANIMAL_START_THRESHOLD",  self.ANIMAL_START_THRESHOLD),
            "ANIMAL_STEP_REDUCTION":   c("ANIMAL_STEP_REDUCTION",   self.ANIMAL_STEP_REDUCTION),
            "ANIMAL_MIN_THRESHOLD":    c("ANIMAL_MIN_THRESHOLD",    self.ANIMAL_MIN_THRESHOLD),
            "VEHICLE_START_THRESHOLD": c("VEHICLE_START_THRESHOLD", self.VEHICLE_START_THRESHOLD),
            "VEHICLE_STEP_REDUCTION":  c("VEHICLE_STEP_REDUCTION",  self.VEHICLE_STEP_REDUCTION),
            "VEHICLE_MIN_THRESHOLD":   c("VEHICLE_MIN_THRESHOLD",   self.VEHICLE_MIN_THRESHOLD),

            # Pesi fake per camera (sovrascrive completamente se presente)
            "FAKE_WEIGHTS": c("FAKE_WEIGHTS", {}),
        }
