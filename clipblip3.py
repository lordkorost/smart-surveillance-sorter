import json
from pathlib import Path
from PIL import Image
import torch
from pathlib import Path
#import tokenize
import torch
from open_clip import create_model_and_transforms
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from open_clip import create_model_and_transforms, tokenize


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Cartella contenente crop e frame ---
folder = Path("2026-02-13/clip")
images = sorted(list(folder.glob("*.jpg")))

# --- Modello CLIP ---
clip_model, preprocess, _ = create_model_and_transforms('ViT-L-14', pretrained='openai')
#clip_model, preprocess, _ = create_model_and_transforms('ViT-L-14', pretrained='openai')
clip_model.to(DEVICE)
clip_model.eval()

# --- Modello BLIP ---
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)
blip_model.eval()

# --- Carica impostazioni ---
with open("config/settings.json") as f:
    settings = json.load(f)

with open("config/prompts.json") as f:
    prompts = json.load(f)

MAIN_CLASSES = prompts["MAIN_CLASSES"]
FAKE_KEYS = prompts["FAKE_CLASSES"]
CLASS_PROMPTS = prompts["CLASS_PROMPTS"]

priority_hierarchy = settings["classification_settings"]["priority_hierarchy"]
weights = settings["classification_settings"]["final_score_weights"]
VETO_THRESHOLD = settings["classification_settings"]["fake_threshold"]

# --- Funzioni ---
def get_clip_score(image_tensor, texts):
    with torch.no_grad():
        image_features = clip_model.encode_image(image_tensor)
        text_tokens = tokenize(texts).to(DEVICE)
        text_features = clip_model.encode_text(text_tokens)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return {cls: float(similarity[0, i]) for i, cls in enumerate(texts)}

# --- Loop su immagini ---
for img_path in images:
    img = Image.open(img_path).convert("RGB")
    crop_img = preprocess(img).unsqueeze(0).to(DEVICE)

    # CLIP score
    clip_scores_crop = get_clip_score(crop_img, MAIN_CLASSES)
    final_scores = {cls: weights["crop"] * clip_scores_crop.get(cls,0) for cls in MAIN_CLASSES}

    # BLIP
    blip_inputs = blip_processor(images=img, return_tensors="pt").to(DEVICE)
    blip_ids = blip_model.generate(**blip_inputs)
    caption = blip_processor.decode(blip_ids[0], skip_special_tokens=True)

    blip_scores = {}
    for cls in MAIN_CLASSES:
        blip_scores[cls] = 0.1 if any(k.lower() in caption.lower() for k in CLASS_PROMPTS[cls]) else 0
        final_scores[cls] += weights["blip"] * blip_scores[cls]

    # Fake classes
    fake_scores = []
    for fk, descs in FAKE_KEYS.items():
        fake_scores.append(max([get_clip_score(crop_img, [d])[d] for d in descs]))
    max_fake_score = max(fake_scores) if fake_scores else 0

    # Priorità gerarchica
    #best_class = "OTHER" if max_fake_score > VETO_THRESHOLD else max(final_scores, key=final_scores.get)
    # Scelta finale con priorità
    if max_fake_score > VETO_THRESHOLD:
        best_class = "OTHER"
    else:
        # controlla la gerarchia: se un livello ha punteggio > 0, scegli quello più alto nella gerarchia
        sorted_classes = sorted(MAIN_CLASSES, key=lambda c: priority_hierarchy.index(c))
        for cls in sorted_classes:
            if final_scores.get(cls, 0) > 0:
                best_class = cls
                break
        else:
            best_class = max(final_scores, key=final_scores.get)

    print(f"{img_path.name} → {best_class}")
