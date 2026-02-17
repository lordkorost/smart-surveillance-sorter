import logging
import time
from pathlib import Path
from collections import defaultdict
from ollama import generate

from smart_surveillance_sorter.constants import CHECKS_DIR, PROMPTS_JSON

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
            log.error(f"❌ Impossibile caricare prompt da: {PROMPTS_JSON}. Raffinamento annullato.")
            self.is_refine = False # Disabilita il refine per evitare crash successivi
            return
        

    def query_vision_model(self, prompt, image_paths):
        vision_model = self.vision_cfg.get("model_name", "qwen3-vl:8b")
        temperature = self.vision_cfg.get("temperature",0.1)
        num_predict = self.vision_cfg.get("num_predict",250)
        top_k = self.vision_cfg.get("top_k",20)
        top_p = self.vision_cfg.get("top_p",0.9)

        if isinstance(image_paths, (str, Path)):
            images = [str(image_paths)]
        else:
            images = [str(p) for p in image_paths]
        #log.debug(prompt)
        try:
            # --- DEBUG LOGS ---
            log.debug(f"📝 PROMPT INVIATO:\n{prompt}")
            # Se image_paths è una lista, stampiamola per sicurezza
            log.debug(f"🖼️ IMMAGINI INVIATE: {image_paths}")
            response = generate(
                model=vision_model,
                prompt=prompt,
                images=images
                # options={
                #     "temperature": temperature,
                #     "num_predict": num_predict,     # LIMITE TOKEN: Se non decide entro ~250 parole, lo taglia
                #     "top_k": top_k,                 # Riduce la casualità "folle"
                #     "top_p": top_p                  # Aiuta a mantenere il ragionamento sensato
                # }
            )
            #log.critical(response)
            # Il trucco per la risposta "nuda e cruda":
            log.critical("--- RAW OLLAMA RESPONSE ---")
            log.critical(repr(response['message']['content']))
            log.critical("---------------------------")
            # thinking_content = ""
            # if hasattr(response, 'thinking') and response.thinking:
            #     thinking_content = response.thinking.strip()
            # answer = response.response.strip().lower()
            # 1. Estrai il thinking (se esiste)
            thinking_content = ""
            if hasattr(response, 'thinking') and response.thinking:
                thinking_content = response.thinking.strip()

            # 2. Prendi l'intera risposta testuale
            full_response = response.response.lower()
            log.info(f"🤔 THINKING:\n{thinking_content}")
            #log.info(f"💬 RISPOSTA COMPLETA:\n{full_response}")
            # # 3. Cerchiamo il verdetto nel testo, anche se sommerso da tag strani
            # if "final verdict:" in full_response:
            #     # Estraiamo solo quello che viene DOPO "final verdict:"
            #     answer = full_response.split("final verdict:")[1].strip()
            #     # Puliamo ulteriormente per prendere solo la prima parola (es: "person")
            #     answer = answer.split()[0].replace('🎯', '').strip()
            # else:
            #     # Fallback: se non c'è il tag formale, cerchiamo le parole chiave nel testo
            #     if "person" in full_response:
            #         answer = "person"
            #     elif "animal" in full_response:
            #         answer = "animal"
            #     elif "vehicle" in full_response:
            #         answer = "vehicle"
            #     else:
            #         answer = "nothing" # O mantieni il verdetto originale di YOLO
            #     # Invece di stampare, ritorniamo tutto
            #     return {
            #         "label": answer,
            #         "thinking": thinking_content
            #     }
            # 3. Estrazione Verdetto
            if "final verdict:" in full_response:
                # Prende quello che c'è dopo "final verdict:"
                answer_part = full_response.split("final verdict:")[1].strip()
                # Prende la prima parola, toglie emoji e anche il punto finale (fondamentale!)
                answer = answer_part.split()[0].replace('🎯', '').replace('.', '').strip()
            else:
                # Fallback se Ollama non usa il tag: prendiamo la prima parola della risposta
                answer = full_response.split()[0].replace('.', '') if full_response.strip() else "nothing"

            return {
                "label": answer,
                "thinking": thinking_content
            }
        except Exception as e:
            log.error(f"Chiamata a Ollama ({vision_model}) fallita: {e}")
            return {"label": "nothing", "thinking": str(e)}

    def refine_single_video(self, video_data):
    
        cam_id = video_data["camera_id"]
        video_path = Path(video_data["video_path"])
        frames = video_data["frames"]
        
        cam_cfg = self.cameras_config.get(cam_id, {})
        has_crops = any(f.get("crop_path") for f in frames)
        
        # Generazione prompt
        prompt = build_dynamic_prompt(
            self.prompts_config, 
            cam_cfg, 
            mode=self.mode, 
            has_crop=has_crops
        )
        log.debug("--- INIZIO PROMPT INVIATO ---")
        log.debug(prompt)
        log.debug("--- FINE PROMPT INVIATO ---")
        scores = defaultdict(float)
        last_frame = {}
        
        # Allineamento con il tuo settings.json ("scoring_system")
        scoring_cfg = self.settings.get("scoring_system", {})
        #weights = scoring_cfg.get("weights", {})
        thresholds = scoring_cfg.get("thresholds", {})

        for frame in frames:
            category = frame["category"]
            conf = frame["confidence"]
            img_path = frame["frame_path"]
            crop_path = frame.get("crop_path")

            # Analisi Multi-Image
            image_inputs = [img_path]
            if crop_path and Path(crop_path).exists():
                image_inputs.append(crop_path)
            
            # vision_answer = self.query_vision_model(prompt, image_inputs)
            result = self.query_vision_model(prompt, image_inputs)
            
            vision_answer = result.get("label", "nothing")
            thinking = result.get("thinking", "")

            # Salviamo il thinking nel record del video per il log finale
            video_data["thinking"] = thinking
            # --- PRIORITÀ PERSONA ---
            if vision_answer == "person":
                log.info(f"🚨 VISION CONFIRMED: Person on {video_path.name}")
                return self._build_result(cam_id, video_path, "person", img_path, frames,thinking)

            # --- SCORING ---
            # Passiamo l'intero blocco scoring_cfg come richiesto dalla tua funzione
            score_gain = calculate_score(category, conf, vision_answer, scoring_cfg)
            scores[category] += score_gain
            last_frame[category] = img_path

            # --- DEBUG PUNTEGGI ---
            score_gain = calculate_score(category, conf, vision_answer, scoring_cfg)
            old_score = scores[category]
            scores[category] += score_gain
            
            log.debug(f"🔍 [DEBUG SCORE] Frame: {Path(img_path).name}")
            log.debug(f"   |-- YOLO: {category} ({conf:.2f})")
            log.debug(f"   |-- VISION: {vision_answer}")
            log.debug(f"   |-- GAIN: {score_gain:.2f} (Weights/Multipliers applied)")
            log.debug(f"   |-- TOTAL {category.upper()} SCORE: {old_score:.2f} -> {scores[category]:.2f}")

            # Bonus Vision Discovery
            if vision_answer in ["animal", "vehicle"] and vision_answer != category:
                scores[vision_answer] += (score_gain * 0.8)
                last_frame[vision_answer] = img_path

            # Early Exit Animal/Vehicle
            if category in ["animal", "vehicle"]:
                target_threshold = thresholds.get(category, 10.0)
                log.debug(f"   |-- CHECK EXIT: {scores[category]:.2f} >= {target_threshold}?")
                if scores[category] >= target_threshold:
                    log.warning(f"🛑 EARLY EXIT TRIGGERED for {category} at score {scores[category]:.2f}")
                    return self._build_result(cam_id, video_path, category, img_path, frames,thinking=thinking)

        return self._run_ballot(scores, frames, last_frame, cam_id, video_path, scoring_cfg,thinking)

    def _run_ballot(self, scores, frames, last_frame, cam_id, video_path, scoring_cfg,thinking):
        
        override_cfg = scoring_cfg.get("yolo_override", {})
        min_conf = override_cfg.get("person_min_conf", 0.58)
        min_score = override_cfg.get("min_total_score_to_skip_override", 1.2)

        active_scores = {k: v for k, v in scores.items() if k in ["person","animal", "vehicle"]}
        
        # Override YOLO Person
        if not active_scores or max(active_scores.values(), default=0) <= min_score:
            high_conf_yolo = [f for f in frames if f["category"] == "person" and f["confidence"] > min_conf]
            if high_conf_yolo:
                best = max(high_conf_yolo, key=lambda x: x["confidence"])
                log.warning(f"⚠️ OVERRIDE: YOLO conferma Person ({best['confidence']:.2f})")
                return self._build_result(cam_id, video_path, "person", best["frame_path"], frames,thinking)

        if active_scores:
            final_cat = max(active_scores, key=active_scores.get)
            if active_scores[final_cat] > min_score:
                return self._build_result(cam_id, video_path, final_cat, last_frame[final_cat], frames,thinking)

        return self._build_result(cam_id, video_path, "nothing", frames[0]["frame_path"] if frames else None, frames,thinking=thinking)

    # def _build_result(self, cam_id, video_path, category, frame_used, all_frames):
    #     video_name = Path(video_path).name
    #     result = {
    #         "camera_id": cam_id,
    #         "video_name": video_name,
    #         "video_path": str(video_path),
    #         "category": category,
    #         "frame_priority": str(frame_used) if frame_used else None, 
    #         "total_frames_scanned": len(all_frames),
    #         "timestamp_analysis": time.strftime("%Y-%m-%d %H:%M:%S"),
    #         "details": all_frames 
    #     }
    #     "camera_id": "03",
    #     "camera_name": "Orto",
    #     "video_name": "NVR_reo_03.mp4",
    #     "video_path": "fails/NVR_reo_03.mp4",
    #     "category": "person",
    #     "confidence": 0.88, # Se Vision, mettiamo 1.0 o la confidenza del frame usato
    #     "best_frame_path": "...", 
    #     "engine": "vision" # o "yolo"
    #     log.info(f"{'✅' if category != 'nothing' else '❌'} Video {video_name} -> {category.upper()}")
    #     return result

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
            
        answer = result.get("label", "nothing")
        thinking = result.get("thinking", "")
        # 4. Validazione e creazione del report
        # Verifichiamo che la risposta sia tra quelle permesse (non 'nothing')
        if answer != "nothing":
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
        
        return {"category": "nothing"}
    


    def analyze_cleanliness(self, image_list, cam_id):
        """
        image_list = [path_riferimento, path_nvr_notturna]
        """
        cam_cfg = self.cameras_config.get(cam_id, {})
        
        # Costruiamo il prompt dal JSON (come abbiamo visto prima)
        from .vision_helpers import build_clean_prompt
        prompt = build_clean_prompt(self.prompts_config, cam_cfg)

        try:
            response = self.query_vision_model(prompt, image_list)
            return response['label']
        except Exception as e:
            return f"error: {str(e)}"

    