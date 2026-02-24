import logging
import torch
from datetime import datetime
from transformers import BlipConfig, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from open_clip import create_model_and_transforms, tokenize
from smart_surveillance_sorter.constants import CLIP_BLIP_JSON
from smart_surveillance_sorter.utils import get_smart_coordinates, is_night_astronomic, load_json

log = logging.getLogger(__name__) 

class ClipBlipEngine:
    def __init__(self, settings, cameras_config, mode,device):
        self.clip_blip_settings = load_json(CLIP_BLIP_JSON)
        self.DEVICE = device
      
        self.mode = mode
       
        self.settings = settings
        self.ch_configs = cameras_config



        # 2. Silenzia i logger interni di Hugging Face
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

        # --- Modello CLIP ---
        # --- Modello CLIP ---
        # Usiamo .get() con dei fallback (valori di default) sensati
        clip_m_name = self.clip_blip_settings.get('clip_model', 'ViT-B/32')
        clip_pre = self.clip_blip_settings.get('clip_pretrained', 'openai')

        self.clip_model, self.preprocess, _ = create_model_and_transforms(
            clip_m_name,
            pretrained=clip_pre
        )
        self.clip_model.to(self.DEVICE)
        self.clip_model.eval()

        # --- Modello BLIP ---
        # Recuperiamo il nome del modello una volta sola
        blip_m_name = self.clip_blip_settings.get('blip_model', 'Salesforce/blip-image-captioning-base')

        # 1. Carichiamo la configurazione
        config = BlipConfig.from_pretrained(blip_m_name)
        config.tie_word_embeddings = False 
        
        # 2. Carichiamo il processor
        self.blip_processor = BlipProcessor.from_pretrained(blip_m_name)

        # 3. Carichiamo il modello
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            blip_m_name, 
            config=config
        ).to(self.DEVICE)
        self.blip_model.eval()
        
        cb_conf = self.clip_blip_settings  # Usiamo cb_conf per evitare conflitti con 'settings'

        self.MAIN_CLASSES = cb_conf.get("MAIN_CLASSES", ["Person", "Others"])
        self.FAKE_KEYS = cb_conf.get("FAKE_KEYS", ["mannequin", "statue", "poster"])
        self.BLIP_KEYWORDS = cb_conf.get("BLIP_KEYWORDS", ["person", "man", "woman", "child"])

        # Soglie di confidenza e significatività
      # Soglie di confidenza e significatività
        self.VETO_THRESHOLD = cb_conf.get("VETO_THRESHOLD", 0.2)
        self.SIGNIFICANCE_THRESHOLD = cb_conf.get("SIGNIFICANCE_THRESHOLD", 0.2)

        # Pesi e Boost
        self.BLIP_BOOST_PERSON = cb_conf.get("BLIP_BOOST_PERSON", 0.35)
        self.BLIP_BOOST_OTHER = cb_conf.get("BLIP_BOOST_OTHER", 0.1)
        self.FINAL_WEIGHT_CROP = cb_conf.get("FINAL_WEIGHT_CROP", 0.7)
        self.FINAL_WEIGHT_FRAME = cb_conf.get("FINAL_WEIGHT_FRAME", 0.3)

        # Logica Anti-Falsi e Priorità
        self.PERSON_PRIORITY_THRESHOLD = cb_conf.get("PERSON_PRIORITY_THRESHOLD", 0.25)
        # self.PERSON_FAKE_RELATIVE = cb_conf.get("PERSON_FAKE_RELATIVE", 1.1)
        
        # ATTENZIONE: Qui DELTA_THRESHOLD deve essere un dizionario di default
        self.DELTA_THRESHOLD = cb_conf.get("DELTA_THRESHOLD", {
            "PERSON": 0.25,
            "ANIMAL": 0.45,
            "VEHICLE": 0.2
        })

        # Parametri Notturni e Animali (Questi erano già quasi tutti corretti)
        self.YOLO_NIGHT_BOOST = cb_conf.get("YOLO_NIGHT_BOOST", 0.3)
        self.PERSON_BOOST_TOLERANCE = cb_conf.get("PERSON_BOOST_TOLERANCE", 0.45)
        self.DAY_ANIMAL_MARGIN = cb_conf.get("DAY_ANIMAL_MARGIN", 0.05)
        self.DAY_ANIMAL_MIN_CONF = cb_conf.get("DAY_ANIMAL_MIN_CONF", 0.20)
        
        # self.ANIMAL_AGG_THRESHOLDS = cb_conf.get("ANIMAL_AGG_THRESHOLDS", {"1": 0.45, "2": 0.40, "default": 0.35})
        # self.VEHICLE_AGG_THRESHOLDS = cb_conf.get("VEHICLE_AGG_THRESHOLDS", {"1": 0.50, "default": 0.40})

        # --- Soglie Dinamiche Animali ---
        self.ANIMAL_START_THRESHOLD = cb_conf.get("ANIMAL_START_THRESHOLD", 0.45)
        self.ANIMAL_STEP_REDUCTION = cb_conf.get("ANIMAL_STEP_REDUCTION", 0.05)
        self.ANIMAL_MIN_THRESHOLD = cb_conf.get("ANIMAL_MIN_THRESHOLD", 0.15)

        # --- Soglie Dinamiche Veicoli ---
        self.VEHICLE_START_THRESHOLD = cb_conf.get("VEHICLE_START_THRESHOLD", 0.50)
        self.VEHICLE_STEP_REDUCTION = cb_conf.get("VEHICLE_STEP_REDUCTION", 0.05)
        self.VEHICLE_MIN_THRESHOLD = cb_conf.get("VEHICLE_MIN_THRESHOLD", 0.20)
      
        # self.priority_hierarchy = ["PERSON", "ANIMAL", "VEHICLE"] #da prendere da settings
        
        # self.lat, self.lon = get_smart_coordinates(self.settings.get("city","Roma"))
        # 1. Recupera il nome della città dai settings
        self.city_name = self.settings.get("city", "Roma")
        
        # 2. Chiama la logica smart (quella che controlla il JSON in config/)
        # Questa funzione restituirà lat/lon dalla cache o ricalcolate
        self.lat, self.lon = get_smart_coordinates(self.city_name)
        
        log.info(f"🚀 Blip start on city={self.city_name} (Lat: {self.lat}, Lon: {self.lon})")
        
        
        # Genera dinamicamente la mappa partendo dai settings di YOLO
        self.label_to_main_class = {}
        groups = self.settings.get("yolo_settings", {}).get("detection_groups", {})

        for main_cls, labels in groups.items():
            for label in labels:
                #if label is None:
                    #print(f"DEBUG CRITICO: Trovata label NONE nel gruppo {main_cls}!")
                self.label_to_main_class[label.lower()] = main_cls

                self.results = {}


    def _get_clip_score(self,image_tensor, texts):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_tensor)
            text_tokens = tokenize(texts).to(self.DEVICE)
            text_features = self.clip_model.encode_text(text_tokens)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            return {cls: float(similarity[0, i]) for i, cls in enumerate(texts)}
        

    def scan_single_video(self,video_data):
        frames = video_data.get("frames", [])
        if not frames:
            log.debug(f"  (No frame found for video={video_data.get('video_path')})")
            return {}
        
        # --- 🚀 RISOLUZIONE ALLA RADICE (FUORI DAL FOR) ---
        is_nvr = video_data.get("resolved_by") == "nvr_image"
        has_yolo_person = any(f.get("category") == "person" for f in frames)
        # Se il video è NVR e YOLO ha trovato persone (anche solo una)
        if is_nvr and has_yolo_person:
            #print(f"  [NVR FAST-TRACK] Video {video_data.get("video_path")} validato istantaneamente via YOLO")
            
            # Prendiamo il primo frame utile per popolare il report
            best_f = next(f for f in frames if f.get("category") == "person")
            
            frame_res = {
                "clip_crop": {"PERSON": 1.0, "ANIMAL": 0.0, "VEHICLE": 0.0},
                "clip_frame": {"PERSON": 1.0, "ANIMAL": 0.0, "VEHICLE": 0.0},
                "blip_caption": f"yolo_nvr_validated_conf_{best_f.get('confidence'):.2f}",
                "blip_scores": {"PERSON": 0, "ANIMAL": 0, "VEHICLE": 0},
                "final_scores": {"PERSON": 1.0, "ANIMAL": 0.0, "VEHICLE": 0.0},
                "fake_scores": {},
                "max_fake_score": 0.0,
                "label": "PERSON",
                "bbox": best_f.get("bbox")  # <--- AGGIUNTA QUI
            }
            
            return {
                video_data.get("video_path"): {
                    "frames": [frame_res],
                    "video_category": "PERSON"
                }
            }
            
        #ignore labels
        camera_id = video_data.get("camera_id")
        cam_config = self.ch_configs.get(camera_id, {})
        ignore_labels = cam_config.get("filters", {}).get("ignore_labels", [])
        # 1. Otteniamo le regole (Custom o Default è trasparente per noi)
        active_rules = self._get_active_blip_rules(camera_id)
        # Identifica quali macro-classi (PERSON, VEHICLE, ANIMAL) disabilitare
        classes_to_ignore = set()
        for label in ignore_labels:
            # if label is None:
            #     print("DEBUG SCANNER: Trovata una label nulla/vuota in ignore_labels!")
            main_cls = self.label_to_main_class.get(label.lower())
            if main_cls:
                classes_to_ignore.add(main_cls)


        frames_list = []
        for frame in frames:
            # 1. Recuperiamo la categoria suggerita da YOLO per questo specifico frame
            yolo_category = frame.get("category", "").upper() # es: "ANIMAL"
            # Qui sta il trucco: chiediamo a CLIP di scegliere tra 
            # la categoria di YOLO e tutte le classi di disturbo (Fake Keys)
            fake_labels = [desc for descs in self.FAKE_KEYS.values() for desc in descs]
            test_labels = [yolo_category] + fake_labels

            
            # usiamo il set completo come fallback, altrimenti usiamo solo quella di YOLO
            current_main_class = [yolo_category] if yolo_category in self.MAIN_CLASSES else self.MAIN_CLASSES
            
            frame_path = frame.get("frame_path")
            crop_path = frame.get("crop_path")
            current_bbox = frame.get("bbox")
            
            # --- Preprocess immagini ---
            img = Image.open(crop_path).convert("RGB")
            crop_img = self.preprocess(img).unsqueeze(0).to(self.DEVICE)
            frame_img = self.preprocess(Image.open(frame_path).convert("RGB")).unsqueeze(0).to(self.DEVICE) if frame_path else crop_img
        
            # --- BLIP caption ---
            blip_inputs = self.blip_processor(images=img, return_tensors="pt").to(self.DEVICE)
            blip_ids = self.blip_model.generate(**blip_inputs)
            caption = self.blip_processor.decode(blip_ids[0], skip_special_tokens=True)

            # --- Calcolo CLIP (Solo sulla classe YOLO!) ---
            clip_scores_crop = self._get_clip_score(crop_img, test_labels)
            clip_scores_frame = self._get_clip_score(frame_img, test_labels)

            # Inizializziamo final_scores solo per le classi presenti in clip_scores
            # Le altre classi (non rilevate da YOLO) rimarranno a 0.0
            final_scores = {cls: 0.0 for cls in self.MAIN_CLASSES}

            # --- Calcolo Score Finali (Crop + Frame) ---
            for cls in current_main_class:
                # Usiamo i pesi dinamici (FINAL_WEIGHT_CROP e FINAL_WEIGHT_FRAME)
                final_scores[cls] = (active_rules["FINAL_WEIGHT_CROP"] * clip_scores_crop.get(cls, 0.0) + 
                                    active_rules["FINAL_WEIGHT_FRAME"] * clip_scores_frame.get(cls, 0.0))
                
                # BLIP boost (Usa BLIP_KEYWORDS e i boost dinamici)
                # Nota: Uso .get(cls, []) per sicurezza sulle keywords
                if any(k.lower() in caption.lower() for k in self.BLIP_KEYWORDS.get(cls, [])):
                    boost = active_rules["BLIP_BOOST_PERSON"] if cls == "PERSON" else active_rules["BLIP_BOOST_OTHER"]
                    final_scores[cls] += boost

            # --- Fake scores (Sempre contro la classe di YOLO) ---
            # Qui usiamo self.FAKE_KEYS (che sono i nomi delle categorie fake)
            fake_prompts = [desc for descs in self.FAKE_KEYS.values() for desc in descs]
            all_prompts = current_main_class + fake_prompts
            all_scores = self._get_clip_score(crop_img, all_prompts)
            
            # fake_scores_dict = {fk: max([all_scores[d] for d in descs]) for fk, descs in self.FAKE_KEYS.items()}
            # max_fake_score = max(fake_scores_dict.values())
            # --- Fake scores pesati ---
            # Prendiamo i pesi dal JSON (default 1.0 se non specificati)
            fake_weights = active_rules.get("FAKE_WEIGHTS", {}) 
            
            # Moltiplichiamo lo score di ogni categoria fake per il suo peso
            fake_scores_dict = {
                fk: max([all_scores[d] for d in descs]) * fake_weights.get(fk, 1.0) 
                for fk, descs in self.FAKE_KEYS.items()
            }
            max_fake_score = max(fake_scores_dict.values())
            # Determiniamo la best_class (Usiamo SIGNIFICANCE_THRESHOLD al posto dello 0.1 fisso)
            # best_class = yolo_category if final_scores[yolo_category] > active_rules["SIGNIFICANCE_THRESHOLD"] else "OTHERS"
            # Se è una persona, usiamo la soglia fissa di significatività per le persone
            current_sig_thresh = self.SIGNIFICANCE_THRESHOLD if yolo_category == "PERSON" else active_rules["SIGNIFICANCE_THRESHOLD"]
            best_class = yolo_category if final_scores[yolo_category] > current_sig_thresh else "OTHERS"

            # Dentro il ciclo video/frame
            dt = datetime.fromisoformat(frame.get("timestamp")) 

            # Verifica dinamica: addio orari fissi!
            is_night = is_night_astronomic(dt, self.lat, self.lon)
            
            if is_night:
                if frame.get("category") == "person":
                    # Usiamo il boost dinamico (aggiungilo al metodo _get_active_blip_rules)
                    final_scores["PERSON"] += active_rules.get("YOLO_NIGHT_BOOST", self.YOLO_NIGHT_BOOST)
                   

            best_score_cls  = max(final_scores, key=final_scores.get)
            best_score_val  = final_scores[best_score_cls]

            # # USIAMO ACTIVE_RULES ANCHE QUI!
            # if final_scores["PERSON"] >= active_rules["PERSON_PRIORITY_THRESHOLD"] and \
            #    max_fake_score < (final_scores["PERSON"] + active_rules["PERSON_BOOST_TOLERANCE"]):
            #     best_class = "PERSON"
            # --- 2. Determinazione Label PERSON (Usa valori self fisssi per stabilità) ---
            # Usiamo self.PERSON_PRIORITY_THRESHOLD invece di active_rules
            # Usiamo self.PERSON_BOOST_TOLERANCE invece di active_rules

            if final_scores["PERSON"] >= self.PERSON_PRIORITY_THRESHOLD and \
            max_fake_score < (final_scores["PERSON"] + self.PERSON_BOOST_TOLERANCE):
                best_class = "PERSON"
            elif(is_night):
                best_class = self._calculate_night_category(best_score_cls, best_score_val, max_fake_score, final_scores, active_rules)
            else:
                best_class = self._calculate_day_category(best_score_cls, best_score_val, max_fake_score, active_rules)

            frame_res = {
                "clip_crop": clip_scores_crop,
                "clip_frame": clip_scores_frame,
                "blip_caption": caption,
                "blip_scores": all_scores, #blip_scores,
                "final_scores": final_scores,
                "fake_scores": fake_scores_dict,
                "max_fake_score": max_fake_score,
                "label": best_class,
                "bbox": current_bbox # <--- AGGIUNGILA QUI!
            }
            frames_list.append(frame_res)      # aggiungi il frame alla lista

        video_dict = {video_data.get("video_path"): { "frames": frames_list }}

        video_cat = self._decide_video_category(video_dict,active_rules)

        # 2. Ottieni il path (la chiave del dizionario)
        video_path = list(video_dict.keys())[0]

        # 3. Aggiungi la categoria dentro il dizionario di quel video
        video_dict[video_path]["video_category"] = video_cat

        return video_dict


    # -------------------------------------------------------------
    # 1) Funzione che decide la categoria per le ore di NOTTE
    def _calculate_night_category(self, best_score_cls, best_score_val, max_fake_score, final_scores: dict, rules: dict) -> str:
        # 1. PRIORITA' PERSON (Calibrata per QuickGELU e notte)
        # # Usiamo rules["PERSON_PRIORITY_THRESHOLD"] (default 0.25)
        # if final_scores["PERSON"] >= rules["PERSON_PRIORITY_THRESHOLD"]:
        #     # Usiamo la tolleranza dinamica (es. 0.45 o 0.50) dal JSON/Default
        #     if max_fake_score < (final_scores["PERSON"] + rules["PERSON_BOOST_TOLERANCE"]):
        #         return "PERSON"
        # 1. PRIORITA' PERSON (Usiamo i valori fissi self)
        if final_scores["PERSON"] >= self.PERSON_PRIORITY_THRESHOLD:
            # Usiamo la tolleranza fissa di sistema
            if max_fake_score < (final_scores["PERSON"] + self.PERSON_BOOST_TOLERANCE):
                return "PERSON"

        # 2. LOGICA PER GLI ALTRI (ANIMAL/VEHICLE)
        # Se il fake score è alto, serve un margine per vincere
        if max_fake_score > rules["VETO_THRESHOLD"]:
            # Recuperiamo il delta specifico per la classe dal dizionario rules
            delta_threshold = rules["DELTA_THRESHOLD"].get(best_score_cls, 0.2)
            if (best_score_val - max_fake_score) > delta_threshold:
                return best_score_cls
            else:
                return "OTHERS"
        
        return best_score_cls
   


    # -------------------------------------------------------------
    # 1) Funzione che decide la categoria per le ore di GUIRNO
    def _calculate_day_category(
        self,
        best_score_cls,
        best_score_val,
        max_fake_score,
        rules: dict # <--- Nuovo parametro
    ) -> str:
        
        # # --- LOGICA PERSONE: NON TOCCATA ---
        # if best_score_cls == "PERSON":
        #     if max_fake_score > rules["VETO_THRESHOLD"]:
        #         delta_threshold = rules["DELTA_THRESHOLD"].get("PERSON", 0.25)
        #         best_class = "PERSON" if (best_score_val - max_fake_score > delta_threshold) else "OTHERS"
        #     else:
        #         best_class = "PERSON"
        #     return best_class
        # --- LOGICA PERSONE: BLINDATA SU SELF ---
        if best_score_cls == "PERSON":
            # Usiamo il VETO_THRESHOLD di sistema per coerenza
            if max_fake_score > self.VETO_THRESHOLD:
                # Usiamo il delta fisso definito nel self.DELTA_THRESHOLD originale
                delta_threshold = self.DELTA_THRESHOLD.get("PERSON", 0.25)
                best_class = "PERSON" if (best_score_val - max_fake_score > delta_threshold) else "OTHERS"
            else:
                best_class = "PERSON"
            return best_class

      
        if best_score_cls == "ANIMAL":
            margin = best_score_val - max_fake_score
            check1 = best_score_val > rules["DAY_ANIMAL_MIN_CONF"]
            check2 = margin > rules["DAY_ANIMAL_MARGIN"]
            if check1 and check2:
                return "ANIMAL"
            else:
                print(f"DEBUG: Cane scartato! Conf OK: {check1} ({best_score_val} > {rules['DAY_ANIMAL_MIN_CONF']}), Margin OK: {check2} ({margin} > {rules['DAY_ANIMAL_MARGIN']})")
                return "OTHERS"
        # --- LOGICA VEICOLI: ---
        if best_score_cls == "VEHICLE":
            delta_threshold = rules["DELTA_THRESHOLD"].get("VEHICLE", 0.2)
            if best_score_val - max_fake_score > delta_threshold:
                return "VEHICLE"
            return "OTHERS"

        return "OTHERS"
  

    def _decide_video_category(self, video_dict, rules: dict):
        video_path = list(video_dict.keys())[0]
        frames = video_dict[video_path]["frames"]
        
        # 1. Check immediato per le persone (Priorità Massima)
        if any(f["label"] == "PERSON" for f in frames):
            return "PERSON"

        # 2. Raggruppiamo gli score per categoria
        animal_scores = [f["final_scores"]["ANIMAL"] for f in frames if f["label"] == "ANIMAL"]
        vehicle_scores = [f["final_scores"]["VEHICLE"] for f in frames if f["label"] == "VEHICLE"]

        #3. ANIMAL
        if animal_scores:
            avg_animal = sum(animal_scores) / len(animal_scores)
            # Calcoliamo la soglia dinamica invece di cercarla nel dizionario
            threshold = self._get_dynamic_threshold(len(animal_scores), "ANIMAL", rules)
            
            if avg_animal > threshold:
                return "ANIMAL"

        # 4. VEHICLE
        if vehicle_scores:
            avg_vehicle = sum(vehicle_scores) / len(vehicle_scores)
            threshold = self._get_dynamic_threshold(len(vehicle_scores), "VEHICLE", rules)
            
            if avg_vehicle > threshold:
                return "VEHICLE"

        return "OTHERS"

        # # 3. Decisione ANIMAL
        # if animal_scores:
        #     avg_animal = sum(animal_scores) / len(animal_scores)
        #     agg_rules = rules.get("ANIMAL_AGG_THRESHOLDS", self.ANIMAL_AGG_THRESHOLDS)
        #     # Soglia basata su quanti frame animali abbiamo trovato
        #     threshold = agg_rules.get(str(len(animal_scores)), agg_rules.get("default", 0.35))
            
        #     if avg_animal > threshold:
        #         return "ANIMAL"

        # # 4. Decisione VEHICLE
        # if vehicle_scores:
        #     avg_vehicle = sum(vehicle_scores) / len(vehicle_scores)
        #     agg_rules = rules.get("VEHICLE_AGG_THRESHOLDS", self.VEHICLE_AGG_THRESHOLDS)
        #     threshold = agg_rules.get(str(len(vehicle_scores)), agg_rules.get("default", 0.40))
            
        #     if avg_vehicle > threshold:
        #         return "VEHICLE"

        
   


    
    def _get_active_blip_rules(self, camera_id):
        """
        Ritorna un dizionario con le regole di raffinamento.
        Sovrascrive i default dell'init solo se presenti nel JSON della telecamera.
        """
        cam_config = self.ch_configs.get(camera_id, {})
        custom = cam_config.get("blip_rules", {})

    

        return {
            # Soglie di base
            "VETO_THRESHOLD": custom.get("VETO_THRESHOLD", self.VETO_THRESHOLD),
            "SIGNIFICANCE_THRESHOLD": custom.get("SIGNIFICANCE_THRESHOLD", self.SIGNIFICANCE_THRESHOLD),
            
            # Pesi per il calcolo final_scores (Crop vs Frame)
            "FINAL_WEIGHT_CROP": custom.get("FINAL_WEIGHT_CROP", self.FINAL_WEIGHT_CROP),
            "FINAL_WEIGHT_FRAME": custom.get("FINAL_WEIGHT_FRAME", self.FINAL_WEIGHT_FRAME),

            # Boost di BLIP (Keywords)
            "BLIP_BOOST_PERSON": custom.get("BLIP_BOOST_PERSON", self.BLIP_BOOST_PERSON),
            "BLIP_BOOST_OTHER": custom.get("BLIP_BOOST_OTHER", self.BLIP_BOOST_OTHER),

            # Logica Persone (Notte e Priorità)
            "YOLO_NIGHT_BOOST": custom.get("YOLO_NIGHT_BOOST", self.YOLO_NIGHT_BOOST),
            "PERSON_PRIORITY_THRESHOLD": custom.get("PERSON_PRIORITY_THRESHOLD", self.PERSON_PRIORITY_THRESHOLD),
            "PERSON_BOOST_TOLERANCE": custom.get("PERSON_BOOST_TOLERANCE", self.PERSON_BOOST_TOLERANCE),

            # Logica Animali e Veicoli (Giorno e Veto)
            "DAY_ANIMAL_MIN_CONF": custom.get("DAY_ANIMAL_MIN_CONF", self.DAY_ANIMAL_MIN_CONF),
            "DAY_ANIMAL_MARGIN": custom.get("DAY_ANIMAL_MARGIN", self.DAY_ANIMAL_MARGIN),
            "DELTA_THRESHOLD": {**self.DELTA_THRESHOLD, **custom.get("DELTA_THRESHOLD", {})},

            # Soglie di Aggregazione Video finale
            # --- NUOVA Logica di Aggregazione Dinamica ---
            # Animali
            "ANIMAL_START_THRESHOLD": custom.get("ANIMAL_START_THRESHOLD", self.ANIMAL_START_THRESHOLD),
            "ANIMAL_STEP_REDUCTION": custom.get("ANIMAL_STEP_REDUCTION", self.ANIMAL_STEP_REDUCTION),
            "ANIMAL_MIN_THRESHOLD": custom.get("ANIMAL_MIN_THRESHOLD", self.ANIMAL_MIN_THRESHOLD),
            
            # Veicoli
            "VEHICLE_START_THRESHOLD": custom.get("VEHICLE_START_THRESHOLD", self.VEHICLE_START_THRESHOLD),
            "VEHICLE_STEP_REDUCTION": custom.get("VEHICLE_STEP_REDUCTION", self.VEHICLE_STEP_REDUCTION),
            "VEHICLE_MIN_THRESHOLD": custom.get("VEHICLE_MIN_THRESHOLD", self.VEHICLE_MIN_THRESHOLD),
            "FAKE_WEIGHTS": custom.get("FAKE_WEIGHTS", {})
        }
    
    def _get_dynamic_threshold(self, count, category, rules):
        prefix = f"{category.upper()}_"
        
        # Valori di default coerenti con il tuo __init__
        default_start = 0.45 if category == "ANIMAL" else 0.50
        
        start = rules.get(f"{prefix}START_THRESHOLD", default_start)
        step = rules.get(f"{prefix}STEP_REDUCTION", 0.05)
        min_t = rules.get(f"{prefix}MIN_THRESHOLD", 0.15)
        
        # Se per errore count è 0, lo trattiamo come 1
        safe_count = max(1, count)
        
        # Calcolo soglia
        threshold = start - (safe_count - 1) * step
        
        return max(min_t, threshold)
