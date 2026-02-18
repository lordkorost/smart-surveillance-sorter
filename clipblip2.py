from collections import defaultdict
import datetime
import json
import os
from pathlib import Path
import time
import torch
from open_clip import create_model_and_transforms
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from open_clip import create_model_and_transforms, tokenize

start_time = time.time()  # tempo iniziale
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Cartella contenente crop e frame ---
folder = Path("2026-02-13/extracted_frames")
images = sorted(list(folder.glob("*.jpg")))

# --- Modello CLIP ---
#clip_model, preprocess, _ = create_model_and_transforms('ViT-L-14', pretrained='openai')
clip_model, preprocess, _ = create_model_and_transforms('ViT-L-14', pretrained='openai')
clip_model.to(DEVICE)
clip_model.eval()

# --- Modello BLIP ---
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)
blip_model.eval()

# --- Classi principali ---
MAIN_CLASSES = ['PERSON','VEHICLE','ANIMAL']

# --- Classi fake / trappola ---
FAKE_KEYS = {
    "SHOE": ["a photo of a shoe or footwear"],
    "WOOD": ["a photo of a piece of wood or a stick"],
    "GARDEN": ["a photo of a garden object or debris"],
    "GROUND": ["a photo of the ground"]
}


BLIP_KEYWORDS = {
    "PERSON": ["person", "people", "human", "officer", "man", "woman", "child","girl", "boy"],
    "VEHICLE": ["car", "truck", "bike", "motorcycle", "bus"],
    "ANIMAL": ["cat", "dog", "horse", "bird", "bear"]
}
VETO_THRESHOLD = 0.2
SIGNIFICANCE_THRESHOLD = 0.2
BLIP_BOOST_PERSON = 0.35      
BLIP_BOOST_OTHER = 0.1
FINAL_WEIGHT_CROP = 0.7
FINAL_WEIGHT_FRAME = 0.3     
PERSON_PRIORITY_THRESHOLD = 0.25  
PERSON_FAKE_RELATIVE = 1.1
DELTA_THRESHOLD = {
        "PERSON": 0.25,
        "ANIMAL": 0.45,
        "VEHICLE": 0.2
    }

import re
from collections import defaultdict, Counter

def extract_video_name_from_frame(frame_path):
    filename = os.path.basename(frame_path)

    # Trova timestamp 14 cifre
    match = re.search(r'\d{14}', filename)
    if not match:
        return None

    timestamp = match.group(0)

    # Prendi tutto fino al timestamp incluso
    prefix = filename.split(timestamp)[0] + timestamp

    return prefix + ".mp4"


def get_clip_score(image_tensor, texts):
    with torch.no_grad():
        image_features = clip_model.encode_image(image_tensor)
        text_tokens = tokenize(texts).to(DEVICE)
        text_features = clip_model.encode_text(text_tokens)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return {cls: float(similarity[0, i]) for i, cls in enumerate(texts)}

results = {}


for img_path in images:
    # Controlla se è crop
    is_crop = "_crop" in img_path.stem
    if not is_crop:
        continue  # processiamo solo dai crop
    base_name = img_path.stem.replace("_crop", "")

    # Trova il frame abbinato
    frame_candidates = [p for p in images if base_name == p.stem and "_crop" not in p.stem]
    frame_path = frame_candidates[0] if frame_candidates else None

    # --- Preprocess immagini ---
    img = Image.open(img_path).convert("RGB")
    crop_img = preprocess(img).unsqueeze(0).to(DEVICE)
    frame_img = preprocess(Image.open(frame_path).convert("RGB")).unsqueeze(0).to(DEVICE) if frame_path else crop_img

 
    # # --- BLIP caption ---
    blip_inputs = blip_processor(images=img, return_tensors="pt").to(DEVICE)
    blip_ids = blip_model.generate(**blip_inputs)
    caption = blip_processor.decode(blip_ids[0], skip_special_tokens=True)

 
    priority_hierarchy = ["PERSON", "ANIMAL", "VEHICLE"]
    

    # --- Calcolo CLIP + BLIP + Fake ---
    clip_scores_crop = get_clip_score(crop_img, MAIN_CLASSES)
    clip_scores_frame = get_clip_score(frame_img, MAIN_CLASSES)
    final_scores = {cls: FINAL_WEIGHT_CROP * clip_scores_crop[cls] + FINAL_WEIGHT_FRAME * clip_scores_frame[cls] for cls in MAIN_CLASSES}

    # BLIP keyword match con boost dinamico
    blip_scores = {cls: 0 for cls in MAIN_CLASSES}
    for cls in MAIN_CLASSES:
        if any(k.lower() in caption.lower() for k in BLIP_KEYWORDS[cls]):
            boost = BLIP_BOOST_PERSON if cls == "PERSON" else BLIP_BOOST_OTHER
            blip_scores[cls] = boost
            final_scores[cls] += boost

    # Fake scores
    all_prompts = MAIN_CLASSES + [desc for descs in FAKE_KEYS.values() for desc in descs]
    all_scores = get_clip_score(crop_img, all_prompts)
    fake_scores_dict = {fk: max([all_scores[d] for d in descs]) for fk, descs in FAKE_KEYS.items()}
    max_fake_score = max(fake_scores_dict.values())


    # --- PARAMETRI NOTTE ---
    NIGHT_HOURS = list(range(0, 7)) + list(range(18, 24))
    YOLO_NIGHT_BOOST = 0.3  # da tarare in base ai test
    YOLO_JSON_PATH = os.path.join(folder, "yolo_scan_res.json")  # cartella immagini

    # estrai ora dal nome del file (dal crop o frame)
    timestamp_str = "20260213184025"  # esempio: prendere dai nomi tipo "NVR reo_00_20260213184025_person_0_crop.jpg"
    dt = datetime.datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
    hour = dt.hour
    # Determiniamo se è giorno per le regole restrittive sugli animali
    is_night = hour in NIGHT_HOURS
    # --- BOOST PERSON NOTTURNO ---
    if hour in NIGHT_HOURS and os.path.exists(YOLO_JSON_PATH):
        with open(YOLO_JSON_PATH, "r") as f:
            yolo_data = json.load(f)
        
        # trova il frame corrente
        file_basename = os.path.basename(img_path)  # es. "NVR reo_00_20260213184025_person_0_crop.jpg"
        yolo_frames = []
        for video in yolo_data:
            for frame in video.get("frames", []):
                if os.path.basename(frame["crop_path"]) == file_basename:
                    yolo_frames.append(frame)
        
        # applica boost se YOLO ha trovato persona affidabile
        for frame in yolo_frames:
            if frame.get("yolo_reliable") and frame.get("category") == "person":
                final_scores["PERSON"] += YOLO_NIGHT_BOOST
                # possiamo anche loggare per debug
                print(f"[NIGHT BOOST] Added {YOLO_NIGHT_BOOST} to PERSON for {file_basename}")

    best_score_cls = max(final_scores, key=final_scores.get)
    best_score_val = final_scores[best_score_cls]

    # --- 1. PRIORITA' PERSON (INTOCCABILE) ---
    # Manteniamo la tua logica: se c'è sospetto persona, vince persona
    if final_scores["PERSON"] >= PERSON_PRIORITY_THRESHOLD and max_fake_score < 0.6:
        best_class = "PERSON"
    
    # --- 2. LOGICA PER TUTTI GLI ALTRI CASI ---
    else:
        # Se siamo di NOTTE, manteniamo la tua logica attuale (0 falsi positivi già confermati)
        if is_night:
            if max_fake_score > VETO_THRESHOLD:
                delta_threshold = DELTA_THRESHOLD.get(best_score_cls, 0.2)
                best_class = best_score_cls if (best_score_val - max_fake_score > delta_threshold) else "OTHER"
            else:
                best_class = best_score_cls
        
        # Se siamo di GIORNO, applichiamo il pugno di ferro sugli ANIMALI
        else:
            if best_score_cls == "ANIMAL":
                # REQUISITI STRINGENTI PER ANIMALE DIURNO:
                # 1. Score animale deve essere molto alto (>0.85)
                # 2. La fake score non deve essere minacciosa (<0.4)
                # 3. Il distacco tra animale e fake deve essere netto (es. 0.45)
                
                distacco_fake = best_score_val - max_fake_score
                
                if best_score_val >= 0.85 and max_fake_score < 0.4 and distacco_fake > 0.45:
                    best_class = "ANIMAL"
                else:
                    # Se non passa questi criteri, è molto probabilmente il tuo legno/scarpa
                    best_class = "OTHER"
            
            # Per le altre classi di giorno (tipo Vehicle o altro) seguiamo la logica standard
            else:
                if max_fake_score > VETO_THRESHOLD:
                    delta_threshold = DELTA_THRESHOLD.get(best_score_cls, 0.2)
                    best_class = best_score_cls if (best_score_val - max_fake_score > delta_threshold) else "OTHER"
                else:
                    best_class = best_score_cls
   
    # best_score_cls = max(final_scores, key=final_scores.get)
    # best_score_val = final_scores[best_score_cls]
    # # --- PRIORITA' PERSON ---
    # if final_scores["PERSON"] >= PERSON_PRIORITY_THRESHOLD and max_fake_score < 0.6:
    #     best_class = "PERSON"
    # else:
    #     if max_fake_score > VETO_THRESHOLD:
    #         # caso speciale PERSON
    #         if best_score_cls == "PERSON" and best_score_val >= PERSON_PRIORITY_THRESHOLD:
    #             if max_fake_score > PERSON_FAKE_RELATIVE * best_score_val:
    #                 best_class = "OTHER"
    #             else:
    #                 best_class = "PERSON"
    #         else:
    #             # logica normale con delta_threshold
    #             delta_threshold = DELTA_THRESHOLD.get(best_score_cls, 0.2)
    #             if best_score_val - max_fake_score > delta_threshold:
    #                 best_class = best_score_cls
    #             else:
    #                 best_class = "OTHER"
    #     else:
          
    #          # --- FILTRO PIÙ SEVERO PER ANIMAL ---
    #         if best_score_cls == "ANIMAL":
    #             if not (final_scores["ANIMAL"] >= 0.75 and max_fake_score < 0.6):
    #                 best_class = "OTHER"
    #             else:
    #                 best_class = "ANIMAL"
    #         else:
    #             # logica originale invariata
    #             significant_classes = [cls for cls, val in final_scores.items() if val > SIGNIFICANCE_THRESHOLD]

    #             if len(significant_classes) > 1:
    #                 non_person_classes = [cls for cls in significant_classes if cls != "PERSON"]
    #                 if len(non_person_classes) > 1:
    #                     sorted_classes = sorted(
    #                         [(cls, final_scores[cls]) for cls in non_person_classes],
    #                         key=lambda x: x[1],
    #                         reverse=True
    #                     )
    #                     best_cls, best_val = sorted_classes[0]
    #                     second_cls, second_val = sorted_classes[1]

    #                     GAP_THRESHOLD = 0.25
    #                     if best_val - second_val > GAP_THRESHOLD:
    #                         best_class = best_cls
    #                     else:
    #                         for cls in priority_hierarchy:
    #                             if cls in non_person_classes:
    #                                 best_class = cls
    #                                 break
    #                 else:
    #                     best_class = non_person_classes[0]
    #             else:
    #                 best_class = best_score_cls
    
    # --- Debug completo
    print(f"Crop: {img_path.name}")
    print(f"  CLIP crop scores: {clip_scores_crop}")
    print(f"  CLIP frame scores: {clip_scores_frame}")
    print(f"  BLIP caption: {caption}")
    print(f"  BLIP scores: {blip_scores}")
    print(f"  FINAL SCORES: {final_scores}")
    print(f"  Fake scores: {fake_scores_dict}")
    print(f"  Max fake score: {max_fake_score} (threshold {VETO_THRESHOLD})")
    print(f"  → Final class: {best_class}\n")

    results[str(img_path)] = {
        "clip_crop": clip_scores_crop,
        "clip_frame": clip_scores_frame,
        "blip_caption": caption,
        "blip_scores": blip_scores,
        "final_scores": final_scores,
        "fake_scores": fake_scores_dict,
        "max_fake_score": max_fake_score,
        "label": best_class
    }


# # --- AGGREGAZIONE VIDEO LEVEL ---

# videos_dict = defaultdict(list)

# # Raggruppa frame per video
# for frame_path, data in results.items():
#     video_name = extract_video_name_from_frame(frame_path)
#     if video_name:
#         videos_dict[video_name].append(data["label"])

# video_categories = {}

# for video_name, labels in videos_dict.items():
#     # PRIORITA' PERSON
#     if "PERSON" in labels:
#         video_categories[video_name] = "PERSON"
#     else:
#         counter = Counter(labels)
#         counter.pop("OTHER", None)
#         if counter:
#             video_categories[video_name] = counter.most_common(1)[0][0]
#         else:
#             video_categories[video_name] = "OTHER"

# # Aggiungiamo al JSON finale
# results["video_categories"] = video_categories


# --- AGGREGAZIONE VIDEO LEVEL PESATA ---

videos_scores = defaultdict(lambda: {"ANIMAL":0, "VEHICLE":0, "count":0})

for frame_path, data in results.items():
    video_name = extract_video_name_from_frame(frame_path)
    if not video_name:
        continue

    if data["label"] == "PERSON":
        videos_scores[video_name]["PERSON_PRESENT"] = True
        continue

    # sommiamo i punteggi finali solo se il frame ha label ANIMAL o VEHICLE
    if data["label"] == "ANIMAL":
        videos_scores[video_name]["ANIMAL"] += data["final_scores"]["ANIMAL"]
        videos_scores[video_name]["count"] += 1
    elif data["label"] == "VEHICLE":
        videos_scores[video_name]["VEHICLE"] += data["final_scores"]["VEHICLE"]
        videos_scores[video_name]["count"] += 1

video_categories = {}

for video_name, scores in videos_scores.items():
    # 1. PRIORITA' PERSON ASSOLUTA
    if scores.get("PERSON_PRESENT", False):
        video_categories[video_name] = "PERSON"
        continue

    # Recuperiamo il count che hai popolato nel primo ciclo
    count = scores["count"]

    if count == 0:
        video_categories[video_name] = "OTHER"
        continue

    # Calcola punteggi medi
    avg_animal = scores["ANIMAL"] / count
    avg_vehicle = scores["VEHICLE"] / count

    # Determiniamo la categoria dominante tra Animal e Vehicle
    if avg_animal >= avg_vehicle:
        # --- LOGICA SEVERA PER ANIMAL ---
        if count == 1:
            # Singolo frame: deve essere quasi perfetto
            video_categories[video_name] = "ANIMAL" if avg_animal > 0.92 else "OTHER"
        elif count == 2:
            video_categories[video_name] = "ANIMAL" if avg_animal > 0.88 else "OTHER"
        else:
            # 3 o più frame: ci fidiamo della persistenza
            video_categories[video_name] = "ANIMAL" if avg_animal > 0.70 else "OTHER"
    
    else:
        # --- LOGICA PER VEHICLE ---
        # Per i veicoli possiamo stare un pelo più bassi (es. 0.85 per frame singolo)
        if count == 1:
            video_categories[video_name] = "VEHICLE" if avg_vehicle > 0.85 else "OTHER"
        else:
            video_categories[video_name] = "VEHICLE" if avg_vehicle > 0.65 else "OTHER"

# for video_name, scores in videos_scores.items():
#     # PRIORITA' PERSON ASSOLUTA
#     if scores.get("PERSON_PRESENT", False):
#         video_categories[video_name] = "PERSON"
#         continue

#     if scores["count"] == 0:
#         video_categories[video_name] = "OTHER"
#         continue

#     # calcola punteggio medio
#     avg_animal = scores["ANIMAL"] / scores["count"]
#     avg_vehicle = scores["VEHICLE"] / scores["count"]

    

    # # scegli il più alto
    # if avg_animal > avg_vehicle:
    #     video_categories[video_name] = "ANIMAL"
    # elif avg_vehicle > avg_animal:
    #     video_categories[video_name] = "VEHICLE"
    # else:
    #     video_categories[video_name] = "OTHER"  # nel caso di pareggio molto basso

results["video_categories"] = video_categories

#results["video_categories"] = video_categories
# Salva i risultati in un file JSON
with open("classification_results.json", "w") as f:
    json.dump(results, f, indent=4)

end_time = time.time()
total_time = end_time - start_time
print(f"\nProcessed {len(images)} images in {total_time:.2f} seconds")
