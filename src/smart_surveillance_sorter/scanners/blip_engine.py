import json
import logging
from pathlib import Path
from .blip_helpers import build_vqa_question
from transformers import BlipForConditionalGeneration, BlipProcessor, BlipForQuestionAnswering
import torch
from PIL import Image
log = logging.getLogger(__name__) 
class BLIPEngine:
    def __init__(self, settings, device, output_dir:Path):
        self.settings = settings
        #self.logger = logger
        self.device = device
        self.output_dir = output_dir
        # 1. Caricamento del Modello (BLIP VQA Base è un ottimo compromesso tra peso e velocità)
        #self.model_name = "Salesforce/blip-vqa-base" 
        #self.model_name = "Salesforce/blip-vqa-capfilt-large"
        #log.info(f"🔄 Caricamento BLIP Engine ({self.model_name}) su {self.device}...")
        # Carica prima la config e imposta il flag a False
        from PIL import Image

        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        # try:
        #     self.processor = BlipProcessor.from_pretrained(self.model_name,use_fast=False)
        #     self.model = BlipForQuestionAnswering.from_pretrained(self.model_name).to(self.device)
        #     log.info("✅ BLIP Engine caricato con successo.")
        # except Exception as e:
        #     log.error(f"❌ Errore nel caricamento di BLIP: {e}")
        #     raise

        # 2. Parametri di Soglia (opzionali, presi dal JSON o default)
        self.threshold = self.settings.get("blip_settings", {}).get("threshold", 0.5)


    def analyze_video_results(self, video_data):
        """
        Analizza i frame con logica di interruzione anticipata (Early Exit).
        Se trova una PERSONA confermata, si ferma e restituisce il risultato.
        """
        detection_groups = self.settings["yolo_settings"]["detection_groups"]
        vqa_evidence = {}

        # 1. ORDINA I FRAME PER PRIORITÀ (Facoltativo ma consigliato)
        # Mettiamo i frame PERSON per primi così attiviamo subito l'early exit
        sorted_frames = sorted(
            video_data["frames"], 
            key=lambda x: x["category"].upper() != "PERSON"
        )

        for frame in sorted_frames:
            category = frame["category"].upper()
            
            # Saltiamo l'analisi se abbiamo già confermato PERSON e questo frame è altro
            # (Anche se con l'ordinamento e il return sotto questo è già coperto)
            
            if category not in vqa_evidence:
                vqa_evidence[category] = {"total": 0, "confirmed": 0}
            
            vqa_evidence[category]["total"] += 1
            
            # Chiamata all'helper e verifica
            ####question = build_vqa_question(category, detection_groups)
            #img_path = frame.get("crop_path") or frame.get("frame_path")
            question = "What is the main object in this image? Answer with one word: person, animal, vehicle, or nothing."
            # --- DOPPIO CONTROLLO: FRAME -> CROP ---
            full_img = frame.get("frame_path")
            crop_img = frame.get("crop_path")
            
            # 1. Tentativo con il frame intero
            is_confirmed, confidence = self._verify_presence(full_img, question)
            
            # 2. Se fallisce, tentiamo con il crop (se esiste)
            #DA RIMETTEREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
            # if not is_confirmed and crop_img:
            #     log.info(f"🔍 Frame intero non convinto. Provo il crop di dettaglio: {Path(crop_img).name}")
            #     is_confirmed, confidence = self._verify_presence(crop_img, question)
            
            
            #is_confirmed, confidence = self._verify_presence(img_path, question)
            
            if is_confirmed:
                vqa_evidence[category]["confirmed"] += 1
                # STAMPA DI TEST 1: Conferma avvenuta
                log.info(f"✨ [CONFIRM_LOG] Categoria: {category} | Confirmed_count ora a: {vqa_evidence[category]['confirmed']}")
                # 🔥 EARLY EXIT: Se è una persona, ci fermiamo qui!
                if category == "PERSON":
                    log.info(f"🏆 Early Exit: {category.upper()} confermata al frame {frame.get('frame_idx', 'N/A')}. Analisi interrotta.")
                    return "person" 
                
                    # Controlliamo se YOLO aveva trovato persone nel video (anche se BLIP ha detto no)
        yolo_saw_person = any(f["category"].upper() == "PERSON" for f in video_data["frames"])
            
            # Se BLIP non ha confermato nessuna persona...
        person_confirmed = vqa_evidence.get("PERSON", {}).get("confirmed", 0) > 0
            
        if yolo_saw_person and not person_confirmed:
            log.info(f"⚠️ YOLO vede una persona ma BLIP non conferma. Sposto in TO_CHECK.")
            return "to_check" # Nuova categoria!
        # STAMPA DI TEST 2: Cosa stiamo mandando al ballottaggio?
        log.info(f"📊 [FINAL_EVIDENCE] Riassunto prima del ballot: {vqa_evidence}")
        # 2. Se arriviamo qui, non abbiamo trovato PERSONE confermate.
        # Chiamiamo il ballottaggio per decidere tra ANIMAL, VEHICLE o EMPTY
        return self._resolve_ballot(video_data, vqa_evidence)
    

    def _verify_presence(self, image_path, question):
        """
        Interroga BLIP per confermare o smentire la presenza di un oggetto.
        Ritorna (bool_confermato, confidence_score).
        """
        # # 1. Recupera tutte le etichette dai gruppi di detection definiti nel JSON
        # detection_groups = self.settings.get("detection_groups", {})
        # # Crea una lista piatta: ['person', 'dog', 'cat', 'bird', ..., 'car', 'truck', ...]
        # all_valid_keywords = [item for sublist in detection_groups.values() for item in sublist]
        # 1. Percorso esatto: settings -> yolo_settings -> detection_groups
        yolo_cfg = self.settings.get("yolo_settings", {})
        groups = yolo_cfg.get("detection_groups", {})

        # 2. Estrazione delle keywords (ora groups non sarà più vuoto!)
        all_valid_keywords = [item for sublist in groups.values() for item in sublist]
        all_valid_keywords.append("yes")
        # all_valid_keywords.append("human")
      
        # Definiamo il file (usa .jsonl per l'append ignorante e sicuro)
        log_file = self.output_dir / "blip_history.jsonl"
        try:
            # # 1. Preparazione Immagine e Input
            # raw_image = Image.open(image_path).convert('RGB')
            # inputs = self.processor(raw_image, question, return_tensors="pt").to(self.device)

            # # 2. Generazione Risposta con punteggi (per calcolare la confidenza)
            # outputs = self.model.generate(
            #     **inputs, 
            #     output_scores=True, 
            #     return_dict_in_generate=True,
            #     max_new_tokens=15
            # )
            # # Poiché return_dict_in_generate=True, outputs è un oggetto complesso che HA .sequences
            # answer = self.processor.decode(outputs.sequences[0], skip_special_tokens=True).lower().strip()
            # 3. Decodifica della risposta testuale
            #log.debug(f"DEBUG BLIP: Risposta='{answer}' | Keywords valide={all_valid_keywords}")

            # Test temporaneo: usa una domanda aperta
            # raw_question = "Is there a person, a human figure or an animal in this image?"
            # inputs_raw = self.processor(raw_image, raw_question, return_tensors="pt").to(self.device)
            # output_raw =self.model.generate(
            #     **inputs_raw, 
            #     output_scores=True, 
            #     return_dict_in_generate=True,
            #     max_new_tokens=15
            # )
            # answer_aperta = self.processor.decode(output_raw.sequences[0], skip_special_tokens=True).lower().strip()
            # log.debug(f"DEBUG RISPOSTA APERTA: '{answer_aperta}'")
            # 4. SALVATAGGIO IMMEDIATO (APPEND)
            # Salviamo Immagine -> Domanda -> Risposta
            inputs = self.processor(image_path, return_tensors="pt")
            outputs = self.model.generate(**inputs)
            print(f"RAW CAPTION: {self.processor.decode(outputs[0], skip_special_tokens=True)}")
            log_entry = {
                "img": str(Path(image_path).name),
                #"q": question,
                "a": outputs
            }

            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
            # 4. Calcolo della confidenza (Softmax sui logit del primo token generato)
            # Prendiamo i punteggi del primo token (dove solitamente c'è "yes" o "no")
            logits = outputs.scores[0]
            probs = torch.softmax(logits, dim=-1)
            confidence = torch.max(probs).item()

            is_confirmed = any(word in outputs for word in all_valid_keywords)
            return is_confirmed, confidence

        except Exception as e:
            log.critical(f"❌ Errore durante verify_presence su {image_path}: {e}")
            return False, 0.0
        

    def _resolve_ballot(self, video_data, vqa_evidence):
        """
        Gestisce i casi in cui BLIP non ha confermato la persona (o non c'era).
        Introduce una clausola di salvaguardia per la confidenza di YOLO.
        """
        
        # 1. CLAUSOLA DI SICUREZZA PER PERSONA (YOLO RELIABILITY)
        # Controlliamo se tra i frame scartati da BLIP ce n'erano alcuni "molto affidabili" per YOLO
        if "PERSON" in vqa_evidence:
            # Se BLIP ha detto NO (confirmed == 0) ma YOLO era sicurissimo
            # Cerchiamo nei frame originali se qualcuno aveva il flag yolo_reliable
            reliable_yolo_person = any(
                f for f in video_data["frames"] 
                if f["category"].upper() == "PERSON" and f.get("yolo_reliable", False)
            )
            
            if reliable_yolo_person:
                log.warning.warning("🛡️ Protezione Person: BLIP ha detto NO, ma YOLO è molto sicuro. Tengo il video come PERSON.")
                return "person"

        # 2. SE PERSONA È DAVVERO ESCLUSA, PROCEDIAMO CON IL RESTO
        if "ANIMAL" in vqa_evidence and vqa_evidence["ANIMAL"]["confirmed"] > 0:
            return "animal"
        
        
            
        if "VEHICLE" in vqa_evidence and vqa_evidence["VEHICLE"]["confirmed"] > 0:
            return "vehicle"
            
        return "empty"