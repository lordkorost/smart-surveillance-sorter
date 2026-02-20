import json
import os

def compare_results(gt_path, res_path):
    # Caricamento file
    with open(gt_path, 'r') as f:
        ground_truth_list = json.load(f)
    with open(res_path, 'r') as f:
        new_results = json.load(f)

    # 1. Normalizziamo il Ground Truth in un dizionario { "nomevideo.mp4": "CATEGORIA" }
    gt_map = {item["video_name"]: item["category"].upper() for item in ground_truth_list}

    # 2. Normalizziamo i Nuovi Risultati { "nomevideo.mp4": "CATEGORIA" }
    # Usiamo os.path.basename per togliere la cartella "2026-02-13/"
    res_map = {os.path.basename(path): data["video_category"].upper() for path, data in new_results.items()}

    categories = ["PERSON", "VEHICLE", "ANIMAL"]
    stats = {cat: {"FP": 0, "FN": 0, "TP": 0} for cat in categories}
    
    # Analizziamo i video presenti in entrambi
    all_videos = set(gt_map.keys()).intersection(set(res_map.keys()))
    
    for video in all_videos:
        actual = gt_map[video]
        predicted = res_map[video]

        for cat in categories:
            if actual == cat and predicted == cat:
                stats[cat]["TP"] += 1
            elif actual == cat and predicted != cat:
                stats[cat]["FN"] += 1
            elif actual != cat and predicted == cat:
                stats[cat]["FP"] += 1

    # Stampa Risultati
    print(f"{'CATEGORIA':<12} | {'TP (OK)':<8} | {'FP (F. Pos)':<12} | {'FN (F. Neg)':<12}")
    print("-" * 55)
    for cat in categories:
        s = stats[cat]
        print(f"{cat:<12} | {s['TP']:<8} | {s['FP']:<12} | {s['FN']:<12}")

    # Dettaglio errori per investigare
    print("\n🔍 DETTAGLIO ERRORI (False Positives):")
    for video in all_videos:
        if gt_map[video] != res_map[video]:
            print(f"Video: {video} | GT: {gt_map[video]} | PREDICTED: {res_map[video]}")

    

if __name__ == "__main__":
    # Inserisci i nomi corretti dei tuoi file
    compare_results("ground_truth.json", "2026-02-13/clip_blip_res.json")