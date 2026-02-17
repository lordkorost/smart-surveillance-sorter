# PROMPTS = {
#     "PERSON": [
#         "a surveillance camera image of a person at night",
#         "a CCTV image of a human"
#     ],
#     "VEHICLE": [
#         "a surveillance camera image of a car",
#         "a surveillance camera image of a truck",
#         "a surveillance camera image of a motorcycle"
#     ],
#     "ANIMAL": [
#         "a surveillance camera image of a dog",
#         "a surveillance camera image of a cat",
#         "a surveillance camera image of an animal at night"
#     ],
#     "OTHER": [
#         "a surveillance camera image of a tree moving in the wind",
#         "a surveillance camera image of a lamppost",
#         "a surveillance camera image of empty outdoor scene at night",
#         "a surveillance camera image of shadows",
#         "a surveillance camera image of ground"
#     ]
# }


import sys
import torch
import open_clip
from PIL import Image
from pathlib import Path
import torch.nn.functional as F

# ======================
# CONFIG
# ======================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD_OTHER = 0.16  # se nessuna classe supera questa soglia → OTHER

# PROMPTS = {
#     "PERSON": [
#         "a full body photo of a person standing outdoors",
#         "a surveillance camera photo of a human",
#     ],
#     "VEHICLE": [
#         "a street photo of a car",
#         "a surveillance camera image of a vehicle",
#         "a photo of a truck on the road",
#     ],
#     "ANIMAL": [
#         "a photo of a cat",
#         "a photo of a dog",
#         "a wild animal outdoors",
#     ],
# }
PROMPTS = {
    "PERSON": [
        "a person in a surveillance camera image",
        "a human visible in a security camera frame"
    ],
    "VEHICLE": [
        "a car in a CCTV image",
        "a truck in a surveillance camera frame"
    ],
    "ANIMAL": [
        "a dog or cat visible in a security camera",
        "a small animal in a surveillance camera frame"
    ]
}

# ======================
# LOAD MODEL
# ======================

print("Loading CLIP model...")
model, preprocess, _ = open_clip.create_model_and_transforms(
    "ViT-L-14",
    pretrained="openai"
)
tokenizer = open_clip.get_tokenizer("ViT-L-14")

model.to(DEVICE)
model.eval()

# ======================
# PRECOMPUTE TEXT EMBEDDINGS
# ======================

with torch.no_grad():
    category_embeddings = {}
    for category, texts in PROMPTS.items():
        tokens = tokenizer(texts).to(DEVICE)
        text_features = model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1)
        # media embedding per categoria
        category_embeddings[category] = text_features.mean(dim=0)

print("Text embeddings ready.\n")

# ======================
# CLASSIFICATION FUNCTION
# ======================

def classify_image(image_path):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features = F.normalize(image_features, dim=-1)

        scores = {}
        for category, text_feat in category_embeddings.items():
            sim = (image_features @ text_feat.unsqueeze(1)).item()
            scores[category] = sim

    # ordinamento per debug
    sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    print(f"RAW scores for {image_path.name}:", sorted_scores)

    # scelta migliore con soglia
    best_class = max(scores, key=scores.get)
    if scores[best_class] < THRESHOLD_OTHER:
        return "OTHER", sorted_scores
    return best_class, sorted_scores

# ======================
# PROCESS FOLDER
# ======================

# def process_folder(folder_path):
#     folder = Path(folder_path)
#     if not folder.exists():
#         print("Folder not found.")
#         return

#     print(f"Processing folder: {folder}\n")
#     results = {}

#     for img_path in folder.glob("*"):
#         if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
#             continue

#         label, scores = classify_image(img_path)
#         results[str(img_path)] = {
#             "label": label,
#             "scores": scores
#         }
#         print(f"{img_path.name:40s} → {label}\n")

#     return results

def process_folder_combined(folder_path):
    folder = Path(folder_path)
    images = list(folder.glob("*.jpg"))
    
    # separa crop e frame
    crops = [img for img in images if img.stem.endswith("_crop")]
    frames = {img.stem: img for img in images if not img.stem.endswith("_crop")}
    
    results = {}

    for crop_path in crops:
        # trova frame corrispondente
        frame_stem = crop_path.stem.replace("_crop", "")
        frame_path = frames.get(frame_stem, None)

        crop_scores = classify_image(crop_path)[1]

        if frame_path:
            frame_scores = classify_image(frame_path)[1]
            # combinazione ponderata
            alpha = 0.7
            final_scores = {k: alpha * crop_scores[k] + (1 - alpha) * frame_scores[k] 
                            for k in crop_scores.keys()}
        else:
            final_scores = crop_scores  # nessun frame disponibile
            alpha = 1.0

        # scelta classe finale
        best_class = max(final_scores, key=final_scores.get)
        if final_scores[best_class] < THRESHOLD_OTHER:
            best_class = "OTHER"

        print(f"{crop_path.name} + {frame_path.name if frame_path else 'no_frame'} → {best_class}")
        print("FINAL SCORES:", final_scores, "\n")

        results[str(crop_path)] = {"label": best_class, "scores": final_scores}

    return results

# ======================
# MAIN
# ======================

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python clip_classifier_v2.py /path/to/crop_folder")
        sys.exit(1)

    process_folder_combined(sys.argv[1])
