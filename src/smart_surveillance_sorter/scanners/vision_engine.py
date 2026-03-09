import logging
import time
from pathlib import Path
from collections import defaultdict
from ollama import generate

from smart_surveillance_sorter.constants import PROMPTS_JSON


from ..utils import calculate_score, load_json
from .vision_helpers import build_dynamic_prompt 
log = logging.getLogger(__name__) 
class VisionEngine:
    """Vision model engine using Ollama for refined object analysis.
    
    Uses large language models to verify and refine YOLO detections through
    natural language reasoning about image content.
    """
    def __init__(self, settings, cameras_config, mode):
        """Initialize Vision engine with Ollama configuration.
        
        Args:
            settings: Application settings
            cameras_config: Camera configurations
            mode: Detection mode
        """
     
        self.settings = settings
        self.cameras_config = cameras_config
        self.mode = mode
     
        self.vision_cfg = self.settings.get("vision_settings", {})
        self.prompts_config = load_json(PROMPTS_JSON)
            
        if not self.prompts_config:
            log.error(f"Is not possible load prompts from file={PROMPTS_JSON}. Refine abort.")
            self.is_refine = False 
            return
        

    def query_vision_model(self, prompt, image_paths):
        """Query the vision model with prompt and images.
        
        Args:
            prompt: Text prompt for the model
            image_paths: Path or list of paths to images
            
        Returns:
            Model response text
        """
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
            

          
         
            # 2. Get response text
            full_response = response.get('response', '').lower().strip()
            thinking_content = response.get('thinking', '') 
            
            
            # 3. Extract verdict with cascade logic
            answer = "others" # Default

            if "final verdict:" in full_response:
                answer_part = full_response.split("final verdict:")[1].strip()
                answer = answer_part.split()[0].strip('.,🎯 \n\t')
            else:
                if "person" in full_response:
                    answer = "person"
                elif "animal" in full_response:
                    answer = "animal"
                elif "vehicle" in full_response or "car" in full_response:
                    answer = "vehicle"
                else:
                    if full_response:
                        answer = full_response.split()[0].strip('.,🎯 \n\t')

            return {
                "label": answer,
                "thinking": thinking_content if thinking_content else "No internal reasoning"
            }
        except Exception as e:
            log.error(f"Call to Ollama mode=({vision_model}) fail error={str(e)}")
            import traceback
            log.error(traceback.format_exc())  # This shows you the exact error line
            return {"label": "others", "thinking": f"Error: {str(e)}"}

    def refine_single_video(self, video_data):
        """Refine YOLO detection results using vision model analysis.
        
        Args:
            video_data: Dictionary with video metadata and detection results
            
        Returns:
            Refined detection results with vision model classifications
        """

        cam_id     = video_data["camera_id"]
        video_path = Path(video_data["video_path"])
        frames     = video_data["frames"]
        cam_cfg    = self.cameras_config.get(cam_id, {})
        scoring_cfg = self.settings.get("scoring_system", {})

        # --- NVR FAST-TRACK ---
        is_nvr          = video_data.get("resolved_by") == "nvr_image"
        has_yolo_person = any(f.get("category") == "person" for f in frames)
        if is_nvr and has_yolo_person:
            return self._build_result(cam_id, video_path, "person",
                                      frames[0]["frame_path"] if frames else None,
                                      frames, thinking="NVR image")

        # Prompt
        prompt = build_dynamic_prompt(
            self.prompts_config,
            cam_cfg,
            mode=self.mode,
            has_crop=False
        )

        scores         = defaultdict(float)
        last_frame     = {}
        confirm_count  = defaultdict(int)   # Counter confirmations per category
        others_count   = 0                  # Counter consecutive others

        for frame in frames:
            category = frame["category"]
            conf     = frame["confidence"]
            img_path = frame["frame_path"]

            result       = self.query_vision_model(prompt, [img_path])
            thinking     = result.get("thinking", "")
            vision_answer = result.get("label", "others").lower()
            if vision_answer == "nothing":
                vision_answer = "others"

            video_data["thinking"] = thinking

            # --- 1. PERSON — exit immediato ---
            if vision_answer == "person":
                return self._build_result(cam_id, video_path, "person",
                                          img_path, frames, thinking)

            # --- 2. ANIMAL / VEHICLE ---
            if vision_answer in ["animal", "vehicle"] and vision_answer == category:
                confirm_count[vision_answer] += 1
                others_count = 0  # reset others
                # Exit se: conf alta (1 conferma) oppure 2 conferme qualsiasi
                if conf >= 0.65 or confirm_count[vision_answer] >= 2:
                    log.debug(f"Fast exit {vision_answer} conf={conf:.2f} confirms={confirm_count[vision_answer]}")
                    return self._build_result(cam_id, video_path, vision_answer,
                                              img_path, frames, thinking)

            # --- 3. OTHERS — 2 volte → bail out al ballot ---
            if vision_answer == "others":
                others_count += 1
                if others_count >= 2:
                    log.debug(f"Fast bail-out: Vision said others {others_count}x → ballot")
                    break

            # Accumulate score for ballot voting
            if vision_answer == "others":
                current_gain = -10.0
            elif vision_answer == category:
                current_gain = max(conf * 2.0, 1.5)
            else:
                current_gain = conf * 0.5

            scores[category]    += current_gain
            last_frame[category] = img_path

        return self._run_ballot(scores, frames, last_frame, cam_id, video_path, scoring_cfg, thinking)

    def _run_ballot(self, scores, frames, last_frame, cam_id, video_path, scoring_cfg, thinking):
        override_cfg = scoring_cfg.get("yolo_override", {})
        min_conf     = override_cfg.get("person_min_conf", 0.58)
        min_score    = override_cfg.get("min_total_score_to_skip_override", 1.2)

        active_scores = {k: v for k, v in scores.items() if k in ["person", "animal", "vehicle"]}
        current_max_score = max(active_scores.values(), default=0)

        # --- STEP 1: OVERRIDE PERSONA ---
        if current_max_score <= min_score:
            high_conf_yolo_person = [f for f in frames
                                     if f["category"] == "person" and f["confidence"] > min_conf]
            if high_conf_yolo_person:
                best = max(high_conf_yolo_person, key=lambda x: x["confidence"])
                log.debug(f"OVERRIDE: YOLO Person ({best['confidence']:.2f}) Vision discarded.")
                return self._build_result(cam_id, video_path, "person",
                                          best["frame_path"], frames, thinking)

        # --- STEP 2: VERDETTO FINALE ---
        if active_scores:
            final_cat = max(active_scores, key=active_scores.get)
            if active_scores[final_cat] > min_score:
                return self._build_result(cam_id, video_path, final_cat,
                                          last_frame[final_cat], frames, thinking)

        # --- STEP 3: DEFAULT ---
        return self._build_result(cam_id, video_path, "others",
                                  frames[0]["frame_path"] if frames else None,
                                  frames, thinking=thinking)

    def _build_result(self, cam_id, video_path, category, frame_used, all_frames,thinking):
        video_path_obj = Path(video_path)
        
        # 1. Retrieve human name from config loaded in Scanner
        # self.cameras_config is the dictionary { "03": {"name": "Garden", ...} }
        cam_info = self.cameras_config.get(str(cam_id), {})
        cam_name = cam_info.get("name", f"Camera_{cam_id}")

        # 2. Search for confidence of used frame
        # Vision does not return confidence, set 1.0 as verdict 
        confidence = 1.0
        if frame_used:
            # Search in details if frame_used exists to get its confidence
            for f in all_frames:
                if f.get("frame_path") == str(frame_used):
                    confidence = f.get("confidence", 1.0)
                    break

        # 3. Build structure identical to YOLO
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
        """Fallback refine for edge cases when primary method fails.
        
        Args:
            suspect: Detection suspect data
            
        Returns:
            Refined classification result
        """
        
        img_path = suspect["image_path"]
        cam_id = suspect["camera_id"]
        cam_id_str = str(cam_id)
        cam_info = self.cameras_config.get(cam_id_str, {})
        cam_name = cam_info.get("name", f"Camera_{cam_id_str}")
       
        prompt = build_dynamic_prompt(
            prompts_config=self.prompts_config, 
            cam_cfg=cam_info,
            mode=self.mode,
            has_crop=False, 
            is_fallback=True
        )

        # 3. Query the model
        result = self.query_vision_model(prompt, [img_path])
     
        answer = result.get("label", "others").lower()
        if answer == "nothing": answer = "others"  # Safety bridge
        thinking = result.get("thinking", "")
        # 4. Validation and report creation
        if answer != "others":
            return {
                "camera_id": cam_id_str,
                "camera_name": cam_name,
                "video_name": Path(suspect["video_path"]).name,
                "video_path": str(suspect["video_path"]),
                "category": answer,
                "confidence": suspect["yolo_data"]["confidence"], # Recuperiamo la confidenza originale
                "best_frame_path": str(img_path),
                "engine": "vision",
                "thinking": thinking, 
            }
        
        return {"category": "others", "thinking": thinking}
    


    def analyze_cleanliness(self, image_list, cam_id,):        
        """Analyze lens cleanliness from a series of frames.
        
        Args:
            image_list: List of image paths to analyze
            cam_id: Camera identifier
            
        Returns:
            Cleanliness assessment result
        """        """
        image_list = [path_riferimento, path_nvr_notturna]
        """
      
        cam_cfg = self.cameras_config.get(cam_id, {})
        vision_model = self.vision_cfg.get("model_name", "qwen3-vl:8b")
        from .vision_helpers import build_clean_prompt
        prompt = build_clean_prompt(self.prompts_config, cam_cfg)
      
        try:
            
            response = generate(
                model=vision_model,
                prompt=prompt,
                images=image_list,
            )
            

            full_response = response.get('response', '').lower().strip()
            thinking_content = response.get('thinking', '') 
           
            log.debug(f"Thinking: {thinking_content}")
           
            # 3. Extract verdict with cascade logic
            answer = "uncertain" # Default

            # Lens-specific logic
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
    

    