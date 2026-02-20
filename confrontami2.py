import json
import os

def compare_results(gt_path, res_path):
    # Caricamento file
    with open(gt_path, 'r') as f:
        ground_truth_list = json.load(f)
    with open(res_path, 'r') as f:
        new_results = json.load(f)

    # 1. Normalizziamo il Ground Truth (Lista -> Dizionario)
    # Usiamo os.path.basename perché nel GT potresti avere percorsi o nomi semplici
    gt_map = {os.path.basename(item["video_name"]): item["category"].upper() for item in ground_truth_list}

    # 2. Normalizziamo i Nuovi Risultati
    # Se il nuovo risultato è una LISTA (come nel tuo esempio):
    if isinstance(new_results, list):
        res_map = {os.path.basename(item["video_name"]): item["category"].upper() for item in new_results}
    # Se invece è un DIZIONARIO { "path": {"video_category": "..."} }:
    else:
        res_map = {os.path.basename(path): data["video_category"].upper() for path, data in new_results.items()}

    # Aggiungiamo NOTHING per beccare i deliri del legno
    categories = ["PERSON", "VEHICLE", "ANIMAL", "NOTHING"]
    stats = {cat: {"FP": 0, "FN": 0, "TP": 0} for cat in categories}
    
    # Analizziamo l'intersezione (video presenti in entrambi)
    all_videos = set(gt_map.keys()).intersection(set(res_map.keys()))
    
    for video in all_videos:
        actual = gt_map[video]
        predicted = res_map[video]

        if actual == predicted:
            if actual in stats:
                stats[actual]["TP"] += 1
        else:
            if actual in stats:
                stats[actual]["FN"] += 1
            if predicted in stats:
                stats[predicted]["FP"] += 1

    # Stampa Tabella
    print(f"\n{'CATEGORIA':<12} | {'TP (OK)':<8} | {'FP (F. Pos)':<12} | {'FN (F. Neg)':<12}")
    print("-" * 55)
    for cat in categories:
        s = stats[cat]
        print(f"{cat:<12} | {s['TP']:<8} | {s['FP']:<12} | {s['FN']:<12}")

    # Investigazione Errori
    print("\n🔍 DETTAGLIO ERRORI (Discrepanze tra GT e Predict):")
    for video in sorted(all_videos):
        if gt_map[video] != res_map[video]:
            print(f"Video: {video:<40} | GT: {gt_map[video]:<8} | PREDICT: {res_map[video]:<8}")

if __name__ == "__main__":
    compare_results("ground_truth.json", "2026-02-13/classification_results.json")