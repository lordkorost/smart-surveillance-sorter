import logging
import torch
from datetime import datetime
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from open_clip import create_model_and_transforms, tokenize
from smart_surveillance_sorter.constants import CAMERAS_JSON, CLIP_BLIP_JSON, SETTINGS_JSON
from smart_surveillance_sorter.utils import get_smart_coordinates, is_night_astronomic, load_json

log = logging.getLogger(__name__) 

class ClipBlipEngine:
    def __init__(self, settings, cameras_config, mode,device):
        self.clip_blip_settings = load_json(CLIP_BLIP_JSON)
        self.DEVICE = device
      
        self.mode = mode
       
        self.settings = settings
        self.ch_configs = cameras_config

        # --- Modello CLIP ---
        self.clip_model, self.preprocess, _ = create_model_and_transforms(
            self.clip_blip_settings['clip_model'],
            pretrained=self.clip_blip_settings['clip_pretrained'])
        self.clip_model.to(self.DEVICE)
        self.clip_model.eval()

        # --- Modello BLIP ---
        self.blip_processor = BlipProcessor.from_pretrained(self.clip_blip_settings['blip_model'])
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            self.clip_blip_settings['blip_model']
            ).to(self.DEVICE)
        self.blip_model.eval()

        # --- Carica i dati delle classi principali
        self.MAIN_CLASSES = self.clip_blip_settings["MAIN_CLASSES"]
        self.FAKE_KEYS = self.clip_blip_settings["FAKE_KEYS"]
        self.BLIP_KEYWORDS = self.clip_blip_settings["BLIP_KEYWORDS"]
        self.VETO_THRESHOLD = self.clip_blip_settings["VETO_THRESHOLD"]
        self.SIGNIFICANCE_THRESHOLD = self.clip_blip_settings["SIGNIFICANCE_THRESHOLD"]
        self.BLIP_BOOST_PERSON = self.clip_blip_settings["BLIP_BOOST_PERSON"]
        self.BLIP_BOOST_OTHER = self.clip_blip_settings["BLIP_BOOST_OTHER"]
        self.FINAL_WEIGHT_CROP = self.clip_blip_settings["FINAL_WEIGHT_CROP"]
        self.FINAL_WEIGHT_FRAME = self.clip_blip_settings["FINAL_WEIGHT_FRAME"]
        self.PERSON_PRIORITY_THRESHOLD = self.clip_blip_settings["PERSON_PRIORITY_THRESHOLD"]
        self.PERSON_FAKE_RELATIVE = self.clip_blip_settings["PERSON_FAKE_RELATIVE"]
        self.DELTA_THRESHOLD = self.clip_blip_settings["DELTA_THRESHOLD"]
        self.YOLO_NIGHT_BOOST = self.clip_blip_settings["YOLO_NIGHT_BOOST"]
        self.NIGHT_HOURS_HOURS = self.clip_blip_settings["NIGHT_HOURS"]
        self.PERSON_BOOST_TOLERANCE = self.clip_blip_settings.get("PERSON_BOOST_TOLERANCE",0.45)
        self.DAY_ANIMAL_MARGIN = self.clip_blip_settings.get("DAY_ANIMAL_MARGIN",0.05)
        self.DAY_ANIMAL_MIN_CONF= self.clip_blip_settings.get("DAY_ANIMAL_MIN_CONF",0.20)
        self.ANIMAL_AGG_THRESHOLDS = self.clip_blip_settings.get("ANIMAL_AGG_THRESHOLDS", {"1": 0.45, "2": 0.40, "default": 0.35})
        self.VEHICLE_AGG_THRESHOLDS = self.clip_blip_settings.get("VEHICLE_AGG_THRESHOLDS", {"1": 0.50, "default": 0.40})
        self.STATIC_MOVEMENT_THRESHOLD = self.clip_blip_settings.get("STATIC_MOVEMENT_THRESHOLD", 25)
        self.NIGHT_HOURS = list(range(
            self.NIGHT_HOURS_HOURS["day_start"],
            self.NIGHT_HOURS_HOURS["sunrise"]+1)) + list(range(
                self.NIGHT_HOURS_HOURS["sunset"],
            self.NIGHT_HOURS_HOURS["midnight"]))
        self.priority_hierarchy = ["PERSON", "ANIMAL", "VEHICLE"] #da prendere da settings
        
        self.lat, self.lon = get_smart_coordinates(self.settings["city"])
        # 1. Recupera il nome della città dai settings
        self.city_name = settings.get("city", "Roma")
        
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
        
    # def scan_videos(self):
    #     # self.yolo_res è la lista di dizionari caricata dal JSON
    #     for video_data in self.yolo_res:
    #         video_dict = self._scan_single_video(video_data)
    #         self.results.update(video_dict)

    #     with open("risultati_finali_clipblip.json", "w") as f:
    #         json.dump(self.results, f, indent=4)

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

            # 1. CONTROLLO STATICO (fuori dal ciclo)
            #is_static = self.check_if_static(frames)
            # Se per qualche motivo la categoria non è tra le nostre MAIN, 
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
            
            for cls in current_main_class:
                final_scores[cls] = (self.FINAL_WEIGHT_CROP * clip_scores_crop.get(cls, 0.0) + 
                                    self.FINAL_WEIGHT_FRAME * clip_scores_frame.get(cls, 0.0))
                #log.debug(f"Classe={cls}, Caption type={type(caption)}, Value={caption}")
                # BLIP boost (solo se la classe è quella attuale)
                if any(k.lower() in caption.lower() for k in self.BLIP_KEYWORDS.get(cls, [])):
                    boost = self.BLIP_BOOST_PERSON if cls == "PERSON" else self.BLIP_BOOST_OTHER
                    #print("BOOST PERSONE BLIP")
                    final_scores[cls] += boost

            # --- Fake scores (Sempre contro la classe di YOLO) ---
            fake_prompts = [desc for descs in self.FAKE_KEYS.values() for desc in descs]
            all_prompts = current_main_class + fake_prompts
            all_scores = self._get_clip_score(crop_img, all_prompts)
            
            fake_scores_dict = {fk: max([all_scores[d] for d in descs]) for fk, descs in self.FAKE_KEYS.items()}
            max_fake_score = max(fake_scores_dict.values())
            
            # Determiniamo la best_class (sarà o quella di YOLO o "OTHER" se gli score sono bassi)
            best_class = yolo_category if final_scores[yolo_category] > 0.1 else "OTHERS"

            # # 1a. parsing ISO‑8601 (Python 3.7+)
            # dt = datetime.fromisoformat(frame.get("timestamp")) 
            # hour = dt.hour                         
            # # 3. verifica
            # is_night = hour in self.NIGHT_HOURS
         
            

            # Dentro il ciclo video/frame
            dt = datetime.fromisoformat(frame.get("timestamp")) 

            # Verifica dinamica: addio orari fissi!
            is_night = is_night_astronomic(dt, self.lat, self.lon)
            #print(f"DEBUG: Video Time: {dt.strftime('%H:%M:%S')} | Località: {self.city_name} | Stato: {is_night}")
            if is_night:
                if frame.get("category") == "person":
                    final_scores["PERSON"] += self.YOLO_NIGHT_BOOST
                    # debug (puoi rimuoverlo se non ti serve)
                    #print(f"[NIGHT BOOST] Added {self.YOLO_NIGHT_BOOST} to PERSON for {frame.get("crop_path")}")

            best_score_cls  = max(final_scores, key=final_scores.get)
            best_score_val  = final_scores[best_score_cls]

            if final_scores["PERSON"] >= self.PERSON_PRIORITY_THRESHOLD and max_fake_score < (final_scores["PERSON"] + self.PERSON_BOOST_TOLERANCE):
                best_class = "PERSON"
            elif(is_night):
                #print(f"DEBUG: Sto per chiamare night_category. Args: frame, {best_score_cls}, {best_score_val}, {max_fake_score}, {type(final_scores)}")
                best_class = self._calculate_night_category(best_score_cls,best_score_val,max_fake_score,final_scores)
            else:
                best_class = self._calculate_day_category(best_score_cls,best_score_val,max_fake_score)

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

        video_cat = self._decide_video_category(video_dict)

        # 2. Ottieni il path (la chiave del dizionario)
        video_path = list(video_dict.keys())[0]

        # 3. Aggiungi la categoria dentro il dizionario di quel video
        video_dict[video_path]["video_category"] = video_cat

        return video_dict


    # -------------------------------------------------------------
    # 1) Funzione che decide la categoria per le ore di NOTTE
    def _calculate_night_category(self,best_score_cls, best_score_val, max_fake_score, final_scores: dict) -> str:
        # 1. PRIORITA' PERSON (Calibrata per QuickGELU e notte)
        # Se YOLO + Boost arrivano a PERSON_PRIORITY_THRESHOLD (0.25)
        if final_scores["PERSON"] >= self.PERSON_PRIORITY_THRESHOLD:
            # Usiamo la tolleranza di +0.45 che abbiamo deciso.
            # In questo caso: 0.40 + 0.45 = 0.85. 
            # Siamo quasi a 0.88! Se vuoi salvarlo proprio tutto, usa +0.50
            if max_fake_score < (final_scores["PERSON"] + 0.50):
                return "PERSON"

        # 2. LOGICA PER GLI ALTRI (ANIMAL/VEHICLE)
        # Se il fake score è alto, serve un margine per vincere
        if max_fake_score > self.VETO_THRESHOLD:
            # Per gli animali useremo un margine piccolo (es. 0.05 o 0.10) come deciso prima
            delta_threshold = self.DELTA_THRESHOLD.get(best_score_cls, 0.2)
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
        max_fake_score
    ) -> str:
        
        # --- LOGICA PERSONE: NON TOCCATA (Torna ai tuoi 189 TP) ---
        if best_score_cls == "PERSON":
            if max_fake_score > self.VETO_THRESHOLD:
                # Usiamo il delta che avevi prima (es. 0.25)
                delta_threshold = self.DELTA_THRESHOLD.get("PERSON", 0.25)
                best_class = "PERSON" if (best_score_val - max_fake_score > delta_threshold) else "OTHERS"
            else:
                best_class = "PERSON"
            return best_class

        # --- LOGICA ANIMALI: CORRETTA PER IL TEST COMPETITIVO ---
        if best_score_cls == "ANIMAL":
            # Qui sta il trucco: nel test competitivo i punteggi sono bassi.
            # Un distacco di 0.05/0.10 è già un segnale forte contro il legno.
            margin = best_score_val - max_fake_score
            
            # Abbassiamo drasticamente il requisito del distacco (da 0.45 a 0.05)
            # e la confidenza base (da 0.70 a 0.20)
            if best_score_val > self.DAY_ANIMAL_MIN_CONF and margin > self.DAY_ANIMAL_MARGIN:
                return "ANIMAL"
            else:
                return "OTHERS"

        # --- LOGICA VEICOLI: (Come prima) ---
        if best_score_cls == "VEHICLE":
            delta_threshold = self.DELTA_THRESHOLD.get("VEHICLE", 0.2)
            if best_score_val - max_fake_score > delta_threshold:
                return "VEHICLE"
            return "OTHERS"

        return "OTHERS"
    

    def _decide_video_category(self, video_dict):
        """
        Prende in input video_dict = { "path/to/video.mp4": { "frames": [...] } }
        Ritorna la stringa della categoria finale.
        """
        # Estraiamo i dati (visto che c'è un solo video nel dict)
        video_path = list(video_dict.keys())[0]
        frames = video_dict[video_path]["frames"]
        
        # Inizializziamo i contatori per questo singolo video
        animal_sum = 0
        vehicle_sum = 0
        count = 0
        person_present = False

        # 1. Ciclo sui frame per accumulare i punteggi
        for frame_data in frames:
            label = frame_data["label"]
            
            if label == "PERSON":
                person_present = True
                # Se c'è una persona, possiamo anche fermarci qui se vuoi
                # ma continuiamo per sicurezza nel caso servissero i log
                continue

            if label == "ANIMAL":
                animal_sum += frame_data["final_scores"]["ANIMAL"]
                count += 1
            elif label == "VEHICLE":
                vehicle_sum += frame_data["final_scores"]["VEHICLE"]
                count += 1

        # 2. Logica di decisione finale
        
        # PRIORITA' PERSON ASSOLUTA
        if person_present:
            return "PERSON"

        if count == 0:
            return "OTHERS"

        # Calcola medie
        avg_animal = animal_sum / count
        avg_vehicle = vehicle_sum / count

        
        # Determiniamo la categoria dominante
        if avg_animal >= avg_vehicle:
            # AGGIUNGI str() QUI SOTTO
            threshold = self.ANIMAL_AGG_THRESHOLDS.get(str(count), self.ANIMAL_AGG_THRESHOLDS.get("default", 0.35))
            
            return "ANIMAL" if avg_animal > threshold else "OTHERS"

        else:
            # E AGGIUNGI str() ANCHE QUI
            threshold = self.VEHICLE_AGG_THRESHOLDS.get(str(count), self.VEHICLE_AGG_THRESHOLDS.get("default", 0.40))
            
            return "VEHICLE" if avg_vehicle > threshold else "OTHERS"



    def _check_if_static(self, frames):
        """
        Ritorna True se l'oggetto è rimasto immobile in tutti i frame.
        """
        if len(frames) < 2:
            return False # Non posso confrontare, assumo sia mobile

        first_bbox = frames[0]['bbox'] # [x1, y1, x2, y2]
        threshold = self.STATIC_MOVEMENT_THRESHOLD
        for i in range(1, len(frames)):
            current_bbox = frames[i]['bbox']
            
            # Calcoliamo quanto si è spostato il centro del box
            # (Usiamo una tolleranza di 15-20 pixel per il rumore di YOLO)
            dx = abs(current_bbox[0] - first_bbox[0])
            dy = abs(current_bbox[1] - first_bbox[1])
            
            if dx > threshold or dy > threshold: 
                return False # Si è mosso! È un animale vero.

        return True # Non si è mai mosso oltre la tolleranza.
