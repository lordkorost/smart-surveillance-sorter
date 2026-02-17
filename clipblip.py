from pathlib import Path
import sys
from open_clip import create_model_and_transforms, tokenize
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD_OTHER = 0.22  # soglia per classificare come OTHER

# CLIP setup
#clip_model, preprocess, _ = create_model_and_transforms('ViT-L-14/openai', pretrained='openai')
clip_model, preprocess, _ = create_model_and_transforms('ViT-L-14', pretrained='openai')

clip_model.to(DEVICE).eval()

# BLIP setup
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model.to(DEVICE).eval()

# Keywords
PERSON_KEYS  = {"person", "people", "human", "man", "woman", "child"}
VEHICLE_KEYS = {"car", "truck", "bus", "bike", "motorcycle"}
ANIMAL_KEYS  = {"cat", "dog", "bird", "horse", "lion", "bear", "elephant",
                "rabbit", "fox", "tiger", "wolf", "zebra"}
OTHER_KEYS   = {"wood", "tree", "lamp", "floor", "fence", "debris", "ground"}


PROMPTS = {
    "PERSON": ["a photo of a person", "a photo of a human"],
    "VEHICLE": ["a photo of a car", "a photo of a truck", "a photo of a bike"],
    "ANIMAL":  ["a photo of a cat", "a photo of a dog", "a photo of a horse",
                "a photo of a wolf", "a photo of a bear", "a photo of a bird"],
}

def classify_clip(image_path):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    text_prompts = [p for ps in PROMPTS.values() for p in ps]
    text_tokens = tokenize(text_prompts).to(DEVICE)
    
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text_tokens)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    # aggrega prompt multipli per classe
    scores = {}
    idx = 0
    for cls, prompts in PROMPTS.items():
        scores[cls] = similarity[0, idx:idx+len(prompts)].mean().item()
        idx += len(prompts)
    return scores

def classify_blip(image_path):
    raw_image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(raw_image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption.lower()

def process_folder_blip_clip(folder_path):
    folder = Path(folder_path)
    images = list(folder.glob("*_crop.jpg"))
    results = {}

    for crop_path in images:
        # CLIP
        clip_scores = classify_clip(crop_path)
        
        # BLIP
        caption = classify_blip(crop_path)
        
        # BLIP keyword check
        blip_scores = {"PERSON":0, "VEHICLE":0, "ANIMAL":0, "OTHER":0}
        if any(word in caption for word in PERSON_KEYS):
            blip_scores["PERSON"] += 0.1
        elif any(word in caption for word in VEHICLE_KEYS):
            blip_scores["VEHICLE"] += 0.1
        elif any(word in caption for word in ANIMAL_KEYS):
            blip_scores["ANIMAL"] += 0.1
        elif any(word in caption for word in OTHER_KEYS):
            blip_scores["OTHER"] += 0.1
        
        # Combina CLIP + BLIP
        final_scores = {k: clip_scores[k] + blip_scores.get(k,0) for k in clip_scores.keys()}

        # Classe finale
        best_class = max(final_scores, key=final_scores.get)
        if final_scores[best_class] < THRESHOLD_OTHER:
            best_class = "OTHER"

        # Stampa debug
        print(f"Crop: {crop_path.name}")
        print(f"  CLIP scores: {clip_scores}")
        print(f"  BLIP caption: {caption}")
        print(f"  BLIP scores: {blip_scores}")
        print(f"  FINAL SCORES: {final_scores}")
        print(f"  → Final class: {best_class}\n")

        results[str(crop_path)] = {
            "clip": clip_scores,
            "blip_caption": caption,
            "blip_scores": blip_scores,
            "final_scores": final_scores,
            "label": best_class
        }

    return results

# Esempio di uso
# results = process_folder_blip_clip("2026-02-13/clip")
# ======================
# MAIN
# ======================

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python clip_classifier_v2.py /path/to/crop_folder")
        sys.exit(1)

    process_folder_blip_clip(sys.argv[1])