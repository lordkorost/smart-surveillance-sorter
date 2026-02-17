import json

def normalize_label(label: str) -> str:
    label = label.lower()
    if label == "others":
        return "other"
    return label

# --- Carica ground truth ---
with open("ground_truth.json", "r") as f:
    ground_truth = json.load(f)

gt_dict = {entry["video_name"]: entry["category"] for entry in ground_truth}

# --- Carica risultati ---
with open("classification_results.json", "r") as f:
    results = json.load(f)

video_results = results["video_categories"]

# --- Inizializza contatori ---
categories = ["person", "animal", "vehicle", "other"]
stats = {cat: {"false_positives": 0, "false_negatives": 0, "mismatches": []} for cat in categories}
total = 0
correct = 0
mismatches = []

# --- Confronto ---
for video_name, predicted in video_results.items():
    predicted = normalize_label(predicted)
    true_label = normalize_label(gt_dict.get(video_name, "UNKNOWN"))
    if true_label == "unknown":
        continue

    total += 1
    if predicted == true_label:
        correct += 1
    else:
        mismatches.append({
            "video_name": video_name,
            "ground_truth": true_label,
            "predicted": predicted
        })
        # Aggiorna falsi positivi e negativi per ogni categoria
        if predicted in categories and predicted != true_label:
            stats[predicted]["false_positives"] += 1
        if true_label in categories and predicted != true_label:
            stats[true_label]["false_negatives"] += 1
        # Mantieni lista dei mismatches per categoria
        if true_label in categories:
            stats[true_label]["mismatches"].append({
                "video_name": video_name,
                "predicted": predicted
            })

# --- Report finale ---
accuracy = correct / total * 100 if total > 0 else 0
print(f"Total videos compared: {total}")
print(f"Correct predictions: {correct}")
print(f"Accuracy: {accuracy:.2f}%\n")

for cat in categories:
    print(f"Category: {cat.upper()}")
    print(f"  False negatives (ground {cat} predicted as other): {stats[cat]['false_negatives']}")
    print(f"  False positives (predicted {cat} but ground other): {stats[cat]['false_positives']}")
    print(f"  Mismatches detail: {len(stats[cat]['mismatches'])} videos\n")

false_negatives_animal = 0
false_positives_animal = 0
mismatches_animal = []

for video_name, predicted in video_results.items():
    predicted = normalize_label(predicted)
    true_label = normalize_label(gt_dict.get(video_name, "UNKNOWN"))
    
    if true_label == "unknown":
        continue

    # Ground ANIMAL ma predetto altro → falso negativo
    if true_label == "animal" and predicted != "animal":
        false_negatives_animal += 1
        mismatches_animal.append({
            "video_name": video_name,
            "ground_truth": true_label,
            "predicted": predicted
        })
    
    # Ground altro ma predetto ANIMAL → falso positivo
    if true_label != "animal" and predicted == "animal":
        false_positives_animal += 1
        mismatches_animal.append({
            "video_name": video_name,
            "ground_truth": true_label,
            "predicted": predicted
        })

print("ANIMAL - False negatives:", false_negatives_animal)
print("ANIMAL - False positives:", false_positives_animal)
print("Mismatches detail:", mismatches_animal)

