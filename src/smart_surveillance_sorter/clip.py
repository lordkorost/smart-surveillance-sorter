import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# --- CONFIGURAZIONE ---
MODEL_NAME = "openai/clip-vit-base-patch32"
IMAGE_DIR = "2026-02-13/clip"  # <--- Cambia questo

# Label da confrontare (puoi aggiungerne altre)
LABELS = [
    "a photo of a person", 
    "a photo of a cat or a dog", 
    "a photo of a vehicle",
    "a photo of a shoe or footwear", # Trappola 1
    "a photo of a piece of wood or a stick", # Trappola 2
    "a photo of a garden object or debris", # Trappola 3
    "a photo of the ground"
]

def run_clip_on_folder():
    # Caricamento modello e processor
    print(f"Loading {MODEL_NAME}...")
    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    # Lista dei file immagine
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    images_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(valid_extensions)]

    if not images_files:
        print("Nessuna immagine trovata nella cartella.")
        return

    print(f"Trovate {len(images_files)} immagini. Inizio analisi...\n")

    for img_name in sorted(images_files):
        img_path = os.path.join(IMAGE_DIR, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
            
            # Prepara gli input per CLIP
            inputs = processor(text=LABELS, images=image, return_tensors="pt", padding=True)

            # Inferenza
            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1) # Normalizza in percentuali

            # Trova la label con probabilità maggiore
            max_idx = torch.argmax(probs, dim=1).item()
            conf = probs[0][max_idx].item() * 100

            print(f"[{img_name}]")
            for i, label in enumerate(LABELS):
                print(f"  - {label}: {probs[0][i].item():.2%}")
            
            print(f"  👉 VERDETTO: {LABELS[max_idx]} ({conf:.1f}%)\n")

        except Exception as e:
            print(f"❌ Errore su {img_name}: {e}")

if __name__ == "__main__":
    run_clip_on_folder()