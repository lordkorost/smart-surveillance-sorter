from pathlib import Path
import torch
from collections import defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
import time
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from open_clip import create_model_and_transforms, tokenize
from smart_surveillance_sorter.constants import CAMERAS_JSON, CLIP_BLIP_JSON, SETTINGS_JSON
from smart_surveillance_sorter.utils import load_json


class ClipBlipEngine:
    def __init__(self):
        self.clip_blip_settings = load_json(CLIP_BLIP_JSON)
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        # --- Cartella contenente crop e frame ---
        self.folder = Path("2026-02-13/extracted_frames") #da passare da scanner
        #self.images = sorted(list(self.folder.glob("*.jpg")))#da leggere da yolo_results
        
        yolo_res_file = self.folder / self.clip_blip_settings['yolo_res_json']
        self.yolo_res = load_json(yolo_res_file)
        self.settings = load_json(SETTINGS_JSON)
        self.ch_configs = load_json(CAMERAS_JSON)

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
        self.NIGHT_HOURS = list(range(
            self.NIGHT_HOURS_HOURS["day_start"],
            self.NIGHT_HOURS_HOURS["sunrise"]+1)) + list(range(
                self.NIGHT_HOURS_HOURS["sunset"],
            self.NIGHT_HOURS_HOURS["midnight"]))
        self.priority_hierarchy = ["PERSON", "ANIMAL", "VEHICLE"] #da prendere da settings
        
        # Genera dinamicamente la mappa partendo dai settings di YOLO
        self.label_to_main_class = {}
        groups = self.settings.get("yolo_settings", {}).get("detection_groups", {})

        for main_cls, labels in groups.items():
            for label in labels:
                self.label_to_main_class[label.lower()] = main_cls

                self.results = {}

            


    def get_clip_score(self,image_tensor, texts):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_tensor)
            text_tokens = tokenize(texts).to(self.DEVICE)
            text_features = self.clip_model.encode_text(text_tokens)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            return {cls: float(similarity[0, i]) for i, cls in enumerate(texts)}
        
    def scan_videos(self):
        # self.yolo_res è la lista di dizionari caricata dal JSON
        for video_data in self.yolo_res:
            # Recupera il percorso del video (opzionale)
            #video_path = video_data.get("video_path", "Nessun path")
            # if video_data.get("resolved_by") != "nvr_image":
            #      print(f"[SKIP] Saltando video non NVR per test coerenza: {video_data.get('video_path')}")
            #      continue
            video_dict = self.scan_single_video(video_data)
            self.results.update(video_dict)

        with open("risultati_finali_clipblip.json", "w") as f:
            json.dump(self.results, f, indent=4)

    def scan_single_video(self,video_data):
        frames = video_data.get("frames", [])
        if not frames:
            print("  (Nessun frame)")
            return {}
        
        # --- 🚀 RISOLUZIONE ALLA RADICE (FUORI DAL FOR) ---
        is_nvr = video_data.get("resolved_by") == "nvr_image"
        has_yolo_person = any(f.get("category") == "person" for f in frames)
        # Se il video è NVR e YOLO ha trovato persone (anche solo una)
        if is_nvr and has_yolo_person:
            print(f"  [NVR FAST-TRACK] Video {video_data.get("video_path")} validato istantaneamente via YOLO")
            
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
            main_cls = self.label_to_main_class.get(label.lower())
            if main_cls:
                classes_to_ignore.add(main_cls)


        frames_list = []
        for frame in frames:
            frame_path = frame.get("frame_path")
            crop_path = frame.get("crop_path")
            current_bbox = frame.get("bbox") # <--- Prendi la bbox da YOLO
            # --- Preprocess immagini ---
            img = Image.open(crop_path).convert("RGB")
            crop_img = self.preprocess(img).unsqueeze(0).to(self.DEVICE)
            frame_img = self.preprocess(Image.open(frame_path).convert("RGB")).unsqueeze(0).to(self.DEVICE) if frame_path else crop_img

        
            # # --- BLIP caption ---
            blip_inputs = self.blip_processor(images=img, return_tensors="pt").to(self.DEVICE)
            blip_ids = self.blip_model.generate(**blip_inputs)
            caption = self.blip_processor.decode(blip_ids[0], skip_special_tokens=True)

            # # --- Calcolo CLIP + BLIP + Fake ---
            # clip_scores_crop = self.get_clip_score(crop_img, self.MAIN_CLASSES)
            # clip_scores_frame = self.get_clip_score(frame_img, self.MAIN_CLASSES)


            # final_scores = {cls: self.FINAL_WEIGHT_CROP * clip_scores_crop[cls] + self.FINAL_WEIGHT_FRAME * clip_scores_frame[cls] for cls in self.MAIN_CLASSES}

            # # BLIP keyword match con boost dinamico
            # blip_scores = {cls: 0 for cls in self.MAIN_CLASSES}
            # for cls in self.MAIN_CLASSES:
            #     if any(k.lower() in caption.lower() for k in self.BLIP_KEYWORDS[cls]):
            #         boost = self.BLIP_BOOST_PERSON if cls == "PERSON" else self.BLIP_BOOST_OTHER
            #         blip_scores[cls] = boost
            #         final_scores[cls] += boost

            # --- Calcolo CLIP + BLIP + Fake ---
            clip_scores_crop = self.get_clip_score(crop_img, self.MAIN_CLASSES)
            clip_scores_frame = self.get_clip_score(frame_img, self.MAIN_CLASSES)

            # --- [NUOVO] FILTRO IGNORE LABELS ---
            # Azzeriamo i punteggi CLIP per le classi da ignorare prima di calcolare il finale
            for cls in classes_to_ignore:
                if clip_scores_crop.get(cls, 0) > 0.1 or clip_scores_frame.get(cls, 0) > 0.1:
                        print(f"  [IGNORE FILTER] Soppressa classe {cls} (Crop: {clip_scores_crop[cls]:.2f}, Frame: {clip_scores_frame[cls]:.2f})")
                if cls in clip_scores_crop:
                    clip_scores_crop[cls] = 0.0
                if cls in clip_scores_frame:
                    clip_scores_frame[cls] = 0.0

            # Calcolo base dei final_scores (con i valori potenzialmente azzerati)
            final_scores = {cls: self.FINAL_WEIGHT_CROP * clip_scores_crop[cls] + self.FINAL_WEIGHT_FRAME * clip_scores_frame[cls] for cls in self.MAIN_CLASSES}

            # BLIP keyword match con boost dinamico
            blip_scores = {cls: 0 for cls in self.MAIN_CLASSES}
            for cls in self.MAIN_CLASSES:
                # Se la classe è tra quelle da ignorare, saltiamo il boost di BLIP
                if cls in classes_to_ignore:
                    continue
                
                if any(k.lower() in caption.lower() for k in self.BLIP_KEYWORDS[cls]):
                    boost = self.BLIP_BOOST_PERSON if cls == "PERSON" else self.BLIP_BOOST_OTHER
                    blip_scores[cls] = boost
                    final_scores[cls] += boost

            # --- IMPORTANTE: Se una classe è ignorata, forziamo il final_score a 0 
            # (utile se CLIP avesse dato punteggi minimi o residui)
            for cls in classes_to_ignore:
                final_scores[cls] = 0.0

            # Fake scores
            all_prompts = self.MAIN_CLASSES + [desc for descs in self.FAKE_KEYS.values() for desc in descs]
            all_scores = self.get_clip_score(crop_img, all_prompts)
            fake_scores_dict = {fk: max([all_scores[d] for d in descs]) for fk, descs in self.FAKE_KEYS.items()}
            max_fake_score = max(fake_scores_dict.values())

            # 1a. parsing ISO‑8601 (Python 3.7+)
            dt = datetime.fromisoformat(frame.get("timestamp")) 
            hour = dt.hour                         
            # 3. verifica
            is_night = hour in self.NIGHT_HOURS


            if is_night:
                if frame.get("yolo_reliable")  and frame.get("category") == "person":
                    final_scores["PERSON"] += self.YOLO_NIGHT_BOOST
                    # debug (puoi rimuoverlo se non ti serve)
                    print(f"[NIGHT BOOST] Added {self.YOLO_NIGHT_BOOST} to PERSON for {frame.get("crop_path")}")

            best_score_cls  = max(final_scores, key=final_scores.get)
            best_score_val  = final_scores[best_score_cls]

            # # --- 1. PRIORITA' PERSON (INTOCCABILE) ---
            # # Manteniamo la tua logica: se c'è sospetto persona, vince persona
            # if final_scores["PERSON"] >= self.PERSON_PRIORITY_THRESHOLD and max_fake_score < 0.6:
            #     best_class = "PERSON"
            # Se YOLO è quasi certo e siamo su NVR, abbassiamo la soglia di sbarramento
            # if is_nvr and frame.get("confidence") > 0.80:
            #     # Se CLIP è cieco, ma non c'è un veto pesante (fake score alto)
            #     if max_fake_score < 0.5:
            #         print(f"DEBUG: YOLO (conf {frame.get('confidence')}) forza PERSON su NVR")
            #         best_class = "PERSON"
            # else:
                # Invece di: max_fake_score < 0.6
                # Usiamo un margine dinamico: il fake deve "battere" la persona per invalidarla
            if final_scores["PERSON"] >= self.PERSON_PRIORITY_THRESHOLD and max_fake_score < (final_scores["PERSON"] + 0.2):
                    best_class = "PERSON"
            elif(is_night):
                    print(f"DEBUG: Sto per chiamare night_category. Args: frame, {best_score_cls}, {best_score_val}, {max_fake_score}, {type(final_scores)}")
                    best_class = self.calculate_night_category(frame,best_score_cls,best_score_val,max_fake_score,final_scores)
            else:
                best_class = self.calculate_day_category(best_score_cls,best_score_val,max_fake_score,final_scores)

            frame_res = {
                "clip_crop": clip_scores_crop,
                "clip_frame": clip_scores_frame,
                "blip_caption": caption,
                "blip_scores": blip_scores,
                "final_scores": final_scores,
                "fake_scores": fake_scores_dict,
                "max_fake_score": max_fake_score,
                "label": best_class,
                "bbox": current_bbox # <--- AGGIUNGILA QUI!
            }
            frames_list.append(frame_res)      # aggiungi il frame alla lista

        video_dict = {video_data.get("video_path"): { "frames": frames_list }}

        video_cat = self.decide_video_category(video_dict)

        # 2. Ottieni il path (la chiave del dizionario)
        video_path = list(video_dict.keys())[0]

        # 3. Aggiungi la categoria dentro il dizionario di quel video
        video_dict[video_path]["video_category"] = video_cat

        # Ora video_dict sarà:
        # {
        #   "path/video.mp4": {
        #       "frames": [...],
        #       "video_category": "ANIMAL"
        #   }
        # }

        return video_dict


    # -------------------------------------------------------------
    # 1) Funzione che decide la categoria per le ore di NOTTE
    # -------------------------------------------------------------
    def calculate_night_category(self,frame,best_score_cls,best_score_val,max_fake_score,final_scores: dict) -> str:
        """
        Restituisce la classe finale quando `hour` rientra in NIGHT_HOURS.
        La logica interna è identica a quella che avevi già scritto.
        """
        
        # if frame.get("yolo_reliable") and frame.get("category") == "person":
        #     final_scores["PERSON"] += self.YOLO_NIGHT_BOOST
        #     # debug (puoi rimuoverlo se non ti serve)
        #     print(f"[NIGHT BOOST] Added {self.YOLO_NIGHT_BOOST} to PERSON for {frame.get("crop_path")}")

        # ---------------------------------------------------------
        # 2) Logica di “selezione” per le ore di NOTTE
        # ---------------------------------------------------------
        # 1. PRIORITA' PERSON (INTOCCABILE)
        # if final_scores["PERSON"] >= self.PERSON_PRIORITY_THRESHOLD and max_fake_score < 0.6:
        #     return "PERSON"


        if max_fake_score > self.VETO_THRESHOLD:
            delta_threshold = self.DELTA_THRESHOLD.get(best_score_cls, 0.2)
            best_class = best_score_cls if (best_score_val - max_fake_score > delta_threshold) else "OTHER"
        else:
            best_class = best_score_cls

        return best_class



    # -------------------------------------------------------------
    # 1) Funzione che decide la categoria per le ore di GUIRNO
    # -------------------------------------------------------------
    def calculate_day_category(
            self,
            best_score_cls,
            best_score_val,
            max_fake_score,
            final_scores: dict,
        ) -> str:
        
        if best_score_cls == "ANIMAL":
            # REQUISITI PIÙ EQUILIBRATI:
            # Abbassiamo lo score minimo a 0.50 (se il distacco è forte)
            # o manteniamo un distacco netto dal rumore.
            distacco_fake = best_score_val - max_fake_score
            
            # Se l'animale è chiaro (>0.70) e il terreno è basso, è un animale.
            # Se è più incerto (>0.50) ma il distacco dal fake è enorme (>0.40), lo prendiamo.
            if (best_score_val >= 0.70 and max_fake_score < 0.5) or (distacco_fake > 0.40):
                best_class = "ANIMAL"
            else:
                best_class = "OTHER"
            
        else:
            # Logica standard per Vehicle e Person
            if max_fake_score > self.VETO_THRESHOLD:
                delta_threshold = self.DELTA_THRESHOLD.get(best_score_cls, 0.2)
                best_class = best_score_cls if (best_score_val - max_fake_score > delta_threshold) else "OTHER"
            else:
                best_class = best_score_cls

        return best_class
    


    # def decide_video_category(self, video_dict):
    #     video_path = list(video_dict.keys())[0]
    #     frames = video_dict[video_path]["frames"]
        
    #     # 1. CONTROLLO STATICO (fuori dal ciclo)
    #     is_static = self.check_if_static(frames)
        
    #     person_present = False
    #     max_animal = 0
    #     max_vehicle = 0

    #     for frame_data in frames:
    #         label = frame_data["label"]
    #         scores = frame_data["final_scores"]
            
    #         if label == "PERSON":
    #             person_present = True
    #             break 

    #         # Raccogliamo i massimi punteggi
    #         if label == "ANIMAL":
    #             if scores["ANIMAL"] > max_animal:
    #                 max_animal = scores["ANIMAL"]
            
    #         elif label == "VEHICLE":
    #             if scores["VEHICLE"] > max_vehicle:
    #                 max_vehicle = scores["VEHICLE"]

    #     # 2. LOGICA DI DECISIONE FINALE
    #     if person_present:
    #         return "PERSON"

    #     if max_vehicle > 0.50:
    #         return "VEHICLE"

    #     # Applichiamo il filtro dinamico per l'animale
    #     if max_animal > 0.45: # Abbassiamo la soglia base per recuperare i notturni
    #         if is_static:
    #             print(f"[DEBUG-STATIC] Video {video_path} ignorato: STATIC (Score: {max_animal:.2f})")
    #             # Se è rimasto immobile in 2+ frame, deve essere quasi perfetto
    #             return "ANIMAL" if max_animal > 0.92 else "OTHER"
    #         else:
    #             # Se è un frame singolo o si è mosso, ci fidiamo dello 0.45
    #             return "ANIMAL"

    #     return "OTHER"
    # def calculate_day_category(
    #         self,
    #         best_score_cls,
    #         best_score_val,
    #         max_fake_score,
    #         final_scores: dict,
    #     ) -> str:
    #     """
    #     Restituisce la classe finale quando `hour` è GIORNO.
    #     La logica interna è identica a quella che avevi già scritto.
    #     """
    #     if best_score_cls == "ANIMAL":
    #         # REQUISITI STRINGENTI PER ANIMALE DIURNO:
    #         # 1. Score animale deve essere molto alto (>0.85)
    #         # 2. La fake score non deve essere minacciosa (<0.4)
    #         # 3. Il distacco tra animale e fake deve essere netto (es. 0.45)
                
    #         distacco_fake = best_score_val - max_fake_score
                
    #         if best_score_val >= 0.85 and max_fake_score < 0.4 and distacco_fake > 0.45:
    #             best_class = "ANIMAL"
    #         else:
    #             # Se non passa questi criteri, è molto probabilmente il tuo legno/scarpa
    #             best_class = "OTHER"
            
    #     # Per le altre classi di giorno (tipo Vehicle o altro) seguiamo la logica standard
    #     else:
    #         if max_fake_score > self.VETO_THRESHOLD:
    #             delta_threshold = self.DELTA_THRESHOLD.get(best_score_cls, 0.2)
    #             best_class = best_score_cls if (best_score_val - max_fake_score > delta_threshold) else "OTHER"
    #         else:
    #             best_class = best_score_cls

    #     return best_class
    


    def decide_video_category(self, video_dict):
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
            return "OTHER"

        # Calcola medie
        avg_animal = animal_sum / count
        avg_vehicle = vehicle_sum / count

        # Determiniamo la categoria dominante
        if avg_animal >= avg_vehicle:
            # --- LOGICA SEVERA PER ANIMAL ---
            if count == 1:
                return "ANIMAL" if avg_animal > 0.92 else "OTHER"
            elif count == 2:
                return "ANIMAL" if avg_animal > 0.88 else "OTHER"
            else:
                return "ANIMAL" if avg_animal > 0.70 else "OTHER"
        
        else:
            # --- LOGICA PER VEHICLE ---
            if count == 1:
                return "VEHICLE" if avg_vehicle > 0.85 else "OTHER"
            else:
                return "VEHICLE" if avg_vehicle > 0.65 else "OTHER"

    # def decide_video_category(self, video_dict):
    #     video_path = list(video_dict.keys())[0]
    #     frames = video_dict[video_path]["frames"]
        
    #     person_present = False
    #     max_animal = 0
    #     max_vehicle = 0

    #     for frame_data in frames:
    #         label = frame_data["label"]
    #         scores = frame_data["final_scores"]
            
    #         # PRIORITÀ ASSOLUTA: Se un frame è PERSON, il video è PERSON.
    #         if label == "PERSON":
    #             person_present = True
    #             break 

    #         if label == "ANIMAL":
    #             if scores["ANIMAL"] > max_animal:
    #                 max_animal = scores["ANIMAL"]
            
    #         elif label == "VEHICLE":
    #             if scores["VEHICLE"] > max_vehicle:
    #                 max_vehicle = scores["VEHICLE"]

    #     if person_present:
    #         return "PERSON"

    #     # Se non c'è una persona, cerchiamo il miglior veicolo o animale
    #     if max_vehicle > 0.50:
    #         return "VEHICLE"

    #     if max_animal > 0.60: # Soglia leggermente più alta per via del legno
    #         return "ANIMAL"

    #     return "OTHER"
    

    def check_if_static(self, frames):
        """
        Ritorna True se l'oggetto è rimasto immobile in tutti i frame.
        """
        if len(frames) < 2:
            return False # Non posso confrontare, assumo sia mobile

        first_bbox = frames[0]['bbox'] # [x1, y1, x2, y2]
        
        for i in range(1, len(frames)):
            current_bbox = frames[i]['bbox']
            
            # Calcoliamo quanto si è spostato il centro del box
            # (Usiamo una tolleranza di 15-20 pixel per il rumore di YOLO)
            dx = abs(current_bbox[0] - first_bbox[0])
            dy = abs(current_bbox[1] - first_bbox[1])
            
            if dx > 25 or dy > 25: 
                return False # Si è mosso! È un animale vero.

        return True # Non si è mai mosso oltre la tolleranza.
if __name__ == "__main__":
    t_start = time.time()
    # 1. Inizializza la tua classe
    # Se il costruttore richiede il path del JSON di YOLO, passalo qui
    scanner = ClipBlipEngine()
    
    print("🚀 Avvio scansione video...")
    
    # 2. Fai partire lo scan
    scanner.scan_videos()
    
    # # 3. Salva o stampa i risultati
    # import json
    # with open("risultati_finali.json", "w") as f:
    #     json.dump(risultati, f, indent=4)
        
    print(f"✅ Scansione completata. Risultati salvati in risultati_finali.json")
    t_end = time.time()

    duration = t_end - t_start
    minutes = int(duration // 60)
    seconds = int(duration % 60)

    print(f"\n🚀 Scansione completata!")
    print(f"⏱️ Tempo totale: {minutes}m {seconds}s")
