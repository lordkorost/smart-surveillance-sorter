import logging
import time
from pathlib import Path
from collections import defaultdict
from ollama import generate

from smart_surveillance_sorter.constants import PROMPTS_JSON

# Importiamo le tue utility esterne
# Assicurati che build_dynamic_prompt e calculate_score siano in utils.py
from ..utils import calculate_score, load_json
from .vision_helpers import build_dynamic_prompt 
log = logging.getLogger(__name__) 
class VisionEngine:
    def __init__(self, settings, cameras_config, mode):
     
        self.settings = settings
        self.cameras_config = cameras_config
        self.mode = mode
        #self.logger = logger

        #self.vision_model=self.settings.get("vision_settings", {}).get("model_name", "qwen2.5-vl")
        self.vision_cfg = self.settings.get("vision_settings", {})
        self.prompts_config = load_json(PROMPTS_JSON)
            
        if not self.prompts_config:
            log.error(f"Is not possible load prompts from file={PROMPTS_JSON}. Refine abort.")
            self.is_refine = False # Disabilita il refine per evitare crash successivi
            return
        

    def query_vision_model(self, prompt, image_paths):
        vision_model = self.vision_cfg.get("model_name", "qwen3-vl:8b")
        temperature = self.vision_cfg.get("temperature",1)
        top_k = self.vision_cfg.get("top_k",20)
        top_p = self.vision_cfg.get("top_p",0.9)

        if isinstance(image_paths, (str, Path)):
            images = [str(image_paths)]
        else:
            images = [str(p) for p in image_paths]
        try:
            response = generate(
                model=vision_model,
                prompt=prompt,
                images=images,
                options = {
                     'temperature': temperature,
                     'top_p': top_p,
                     'top_k': top_k
                 }
            )
            

          
         
            # 2. Prendi il testo (già in minuscolo)
            full_response = response.get('response', '').lower().strip()
            thinking_content = response.get('thinking', '') 
            
            #log.info(f"🤔 FULL TEXT:\n{full_response}")
              # --- NUOVA LOGICA DI ESTRAZIONE ---
            
            # 3. Estrazione Verdetto con logica a cascata
            answer = "others" # Default

            if "final verdict:" in full_response:
                answer_part = full_response.split("final verdict:")[1].strip()
                answer = answer_part.split()[0].strip('.,🎯 \n\t')
            else:
                # Se Ollama è pigro e risponde solo "person" o "animal"
                if "person" in full_response:
                    answer = "person"
                elif "animal" in full_response:
                    answer = "animal"
                elif "vehicle" in full_response or "car" in full_response:
                    answer = "vehicle"
                else:
                    # Ultima spiaggia: prendi la prima parola della risposta
                    if full_response:
                        answer = full_response.split()[0].strip('.,🎯 \n\t')

            # ORA lo stampiamo fuori dai blocchi IF, così lo vedi SEMPRE
            #log.critical(f"🎯 VERDETTO ESTRATTO: {answer}")

            return {
                "label": answer,
                "thinking": thinking_content if thinking_content else "No internal reasoning"
            }
        except Exception as e:
            log.error(f"Call to Ollama mode=({vision_model}) fail error={str(e)}")
            import traceback
            log.error(traceback.format_exc()) # Questo ti dice l'esatta riga dell'errore
            return {"label": "others", "thinking": f"Error: {str(e)}"}

    def refine_single_video(self, video_data):

    
        cam_id = video_data["camera_id"]
        video_path = Path(video_data["video_path"])
        frames = video_data["frames"]
        
        cam_cfg = self.cameras_config.get(cam_id, {})
        #has_crops = any(f.get("crop_path") for f in frames)
        has_crops = False
        # Generazione prompt
         # --- 🚀 RISOLUZIONE ALLA RADICE (FUORI DAL FOR) ---
        is_nvr = video_data.get("resolved_by") == "nvr_image"
        has_yolo_person = any(f.get("category") == "person" for f in frames)
        # Se il video è NVR e YOLO ha trovato persone (anche solo una)
        if is_nvr and has_yolo_person:
            #print(f"  [NVR FAST-TRACK] Video {video_data.get("video_path")} validato istantaneamente via YOLO")
            # Prendiamo il primo frame utile per popolare il report
            #best_f = next(f for f in frames if f.get("category") == "person")
            category = "person"
            return self._build_result(cam_id, video_path, "person", frames[0]["frame_path"] if frames else None, frames, thinking="NVR image")

        prompt = build_dynamic_prompt(
            self.prompts_config, 
            cam_cfg, 
            mode=self.mode, 
            has_crop=has_crops
        )
        # print("--- INIZIO PROMPT INVIATO ---")
        # print(prompt)
        # print("--- FINE PROMPT INVIATO ---")
        scores = defaultdict(float)
        last_frame = {}
        
        # Allineamento con il tuo settings.json ("scoring_system")
        scoring_cfg = self.settings.get("scoring_system", {})
        #weights = scoring_cfg.get("weights", {})
        #thresholds = scoring_cfg.get("thresholds", {})

        for frame in frames:
            category = frame["category"]
            conf = frame["confidence"]
            img_path = frame["frame_path"]
            #crop_path = frame.get("crop_path")

            # Analisi Multi-Image
            image_inputs = [img_path]
            # if crop_path and Path(crop_path).exists():
            #     image_inputs.append(crop_path)
            #print(category,conf,image_inputs)
            # vision_answer = self.query_vision_model(prompt, image_inputs)
            result = self.query_vision_model(prompt, image_inputs)
            #print(result)
            #vision_answer = result.get("label", "nothing")
            thinking = result.get("thinking", "")
            vision_answer = result.get("label", "others").lower()
            if vision_answer == "nothing": vision_answer = "others" # Bridge per sicurezza    
            # Salviamo il thinking nel record del video per il log finale
            video_data["thinking"] = thinking
            # --- PRIORITÀ PERSONA ---
            # --- LOGICA DENTRO IL CICLO FRAME ---

            # 1. Se Vision dice Persona, chiudiamo subito (Massima Priorità)
            if vision_answer == "person":
                return self._build_result(cam_id, video_path, "person", img_path, frames, thinking)

            # 2. Calcolo del guadagno (Gain)
            # Usiamo una logica di smentita aggressiva per eliminare il legno
            if vision_answer == "others":
                # Se Vision non vede nulla, diamo un peso negativo che annulla la confidenza di YOLO
                current_gain = -10.0 
            elif vision_answer == category:
                # Se Vision conferma (animal==animal o vehicle==vehicle), bonus!
                current_gain = max(conf * 2.0, 1.5) # Forza almeno a 1.5 se c'è accordo
            else:
                # Se c'è discordanza (es. YOLO dice Animal, Vision dice Vehicle)
                current_gain = conf * 0.5

            scores[category] += current_gain
            last_frame[category] = img_path

            # 3. Early Exit (Solo per Animal/Vehicle confermati)
            # Se il punteggio sale e Vision è d'accordo, usciamo senza aspettare altri frame
            if category in ["animal", "vehicle"] and scores[category] >= 5.0 and vision_answer == category:
                return self._build_result(cam_id, video_path, category, img_path, frames, thinking)



        return self._run_ballot(scores, frames, last_frame, cam_id, video_path, scoring_cfg,thinking)

    def _run_ballot(self, scores, frames, last_frame, cam_id, video_path, scoring_cfg, thinking):
        override_cfg = scoring_cfg.get("yolo_override", {})
        min_conf = override_cfg.get("person_min_conf", 0.58)
        # Punteggio minimo per fidarsi dei voti accumulati (animali/veicoli)
        min_score = override_cfg.get("min_total_score_to_skip_override", 1.2)

        active_scores = {k: v for k, v in scores.items() if k in ["person", "animal", "vehicle"]}
        
        # --- STEP 1: OVERRIDE PERSONA (Il tuo Veto) ---
        # Se non abbiamo punteggi forti, controlliamo se YOLO aveva visto una persona con alta confidenza
        current_max_score = max(active_scores.values(), default=0)
        
        if current_max_score <= min_score:
            high_conf_yolo_person = [f for f in frames if f["category"] == "person" and f["confidence"] > min_conf]
            if high_conf_yolo_person:
                best = max(high_conf_yolo_person, key=lambda x: x["confidence"])
                log.debug(f"⚠️ OVERRIDE: YOLO confirm Person ({best['confidence']:.2f}) Vision category discarded.")
                return self._build_result(cam_id, video_path, "person", best["frame_path"], frames, thinking)

        # --- STEP 2: VERDETTO FINALE ---
        if active_scores:
            final_cat = max(active_scores, key=active_scores.get)
            if active_scores[final_cat] > min_score:
                return self._build_result(cam_id, video_path, final_cat, last_frame[final_cat], frames, thinking)

        # --- STEP 3: DEFAULT ---
        return self._build_result(cam_id, video_path, "others", frames[0]["frame_path"] if frames else None, frames, thinking=thinking)


    def _build_result(self, cam_id, video_path, category, frame_used, all_frames,thinking):
        video_path_obj = Path(video_path)
        
        # 1. Recupero il nome umano dal config caricato nello Scanner
        # self.cameras_config è il dizionario { "03": {"name": "Orto", ...} }
        cam_info = self.cameras_config.get(str(cam_id), {})
        cam_name = cam_info.get("name", f"Camera_{cam_id}")

        # 2. Cerco la confidenza del frame usato (se presente nei dettagli)
        # Se Vision non restituisce confidenza, mettiamo 1.0 come verdetto certo
        confidence = 1.0
        if frame_used:
            # Cerchiamo nei dettagli se esiste il frame_used per prenderne la confidenza
            for f in all_frames:
                if f.get("frame_path") == str(frame_used):
                    confidence = f.get("confidence", 1.0)
                    break

        # 3. Costruisco la struttura identica a YOLO
        return {
            "camera_id": str(cam_id),
            "camera_name": cam_name,
            "video_name": video_path_obj.name,
            "video_path": str(video_path_obj),
            "category": category,
            "confidence": confidence,
            "best_frame_path": str(frame_used) if frame_used else None,
            "engine": "vision",
            "thinking":thinking
        }
    

    def refine_fallback(self, suspect):
        
        img_path = suspect["image_path"]
        cam_id = suspect["camera_id"]
        cam_id_str = str(cam_id)
        cam_info = self.cameras_config.get(cam_id_str, {})
        cam_name = cam_info.get("name", f"Camera_{cam_id_str}")
        # 1. Recuperiamo la configurazione specifica della camera
        #cam_cfg = self.cameras_data.get(str(cam_id), {})

        # 2. Generiamo il prompt dinamico usando il tuo build_dynamic_prompt
        # Passiamo is_fallback=True per attivare il template di "deep scan"
        prompt = build_dynamic_prompt(
            prompts_config=self.prompts_config, # Il tuo JSON dei prompt
            cam_cfg=cam_info,
            mode=self.mode,
            has_crop=False, # Sulle NVR usiamo l'immagine intera
            is_fallback=True
        )

        # 3. Interroghiamo il modello
        #answer = self.query_vision_model(prompt, [img_path])
        result = self.query_vision_model(prompt, [img_path])
            
        #answer = result.get("label", "others")
        answer = result.get("label", "others").lower()
        if answer == "nothing": answer = "others" # Bridge per sicurezza
        thinking = result.get("thinking", "")
        # 4. Validazione e creazione del report
        # Verifichiamo che la risposta sia tra quelle permesse (non 'nothing')
        if answer != "others":
            return {
                "camera_id": cam_id_str,
                "camera_name": cam_name,
                "video_name": Path(suspect["video_path"]).name,
                "video_path": str(suspect["video_path"]),
                "category": answer,
                "confidence": suspect["yolo_data"]["confidence"], # Recuperiamo la confidenza originale
                "best_frame_path": str(img_path),
                "engine": "fallback_nvr",
                "thinking": thinking, 
            }
        
        return {"category": "others"}
    


    def analyze_cleanliness(self, image_list, cam_id,):
        """
        image_list = [path_riferimento, path_nvr_notturna]
        """
      
        cam_cfg = self.cameras_config.get(cam_id, {})
        vision_model = self.vision_cfg.get("model_name", "qwen3-vl:8b")
        # Costruiamo il prompt dal JSON (come abbiamo visto prima)
        from .vision_helpers import build_clean_prompt
        prompt = build_clean_prompt(self.prompts_config, cam_cfg)
      
        try:
            
            response = generate(
                model=vision_model,
                prompt=prompt,
                images=image_list,
            )
            

          
           
            # 2. Prendi il testo (già in minuscolo)
            full_response = response.get('response', '').lower().strip()
            thinking_content = response.get('thinking', '') 
           
           #log.info(f"🤔 FULL TEXT:\n{full_response}")
              # --- NUOVA LOGICA DI ESTRAZIONE ---
            log.debug(f"Thinking: {thinking_content}")
           
            # 3. Estrazione Verdetto con logica a cascata
            answer = "uncertain" # Default

            # Logica specifica per la pulizia
            if any(word in full_response for word in ["clean", "perfect"]):
                answer = "clean"
            elif any(word in full_response for word in ["dirty", "dust"]):
                answer = "dirty"
            else:
                answer = "uncertain"
            
            return answer
        except Exception as e:
            return f"error: {str(e)}"

    

    def resize_for_vision(image_path, output_path, size=(1280, 720)):
        from PIL import Image
        
        with Image.open(image_path) as img:
            # Mantiene le proporzioni se preferisci, o forza il 720p
            img.thumbnail(size, Image.Resampling.LANCZOS)
            img.save(output_path, "JPEG", quality=85)
        return output_path
    

    