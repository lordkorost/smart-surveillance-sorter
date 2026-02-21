import argparse
import json
import os
from pathlib import Path

def compare_results(session_dir=None, gt_file=None, res_file=None, log=None):
    def _print(msg): log.info(msg) if log else print(msg)
    def _err(msg): log.error(msg) if log else print(f"❌ {msg}")

    # --- LOGICA DI RISOLUZIONE PERCORSI ---
    path_gt = None
    path_res = None

    if session_dir:
        s_path = Path(session_dir)
        path_gt = s_path / "ground_truth.json"
        path_res = s_path / "classification_results.json"
    
    # Se l'utente specifica i file esplicitamente, questi vincono sulla directory
    if gt_file:
        path_gt = Path(gt_file)
    if res_file:
        path_res = Path(res_file)

    if not path_gt or not path_res or not path_gt.exists() or not path_res.exists():
        _err(f"File non trovati!\nGT: {path_gt}\nRES: {path_res}")
        _err("Specifica una --dir valida o i percorsi diretti con --gt e --res")
        return

    _print(f"📊 Confronto in corso...\n📖 GT: {path_gt}\n🤖 AI: {path_res}")

    # --- CARICAMENTO E NORMALIZZAZIONE ---
    try:
        with open(path_gt, 'r', encoding='utf-8') as f:
            gt_list = json.load(f)
        with open(path_res, 'r', encoding='utf-8') as f:
            res_data = json.load(f)
    except Exception as e:
        _err(f"Errore caricamento JSON: {e}")
        return

    # Normalizzazione GT
    gt_map = {os.path.basename(item["video_name"]): item["category"].upper() for item in gt_list}

    # Normalizzazione Risultati AI
    if isinstance(res_data, list):
        res_map = {os.path.basename(item["video_name"]): item["category"].upper() for item in res_data}
    else:
        res_map = {os.path.basename(path): data["video_category"].upper() for path, data in res_data.items()}

    # --- CALCOLO STATISTICHE ---
    categories = ["PERSON", "VEHICLE", "ANIMAL", "OTHERS"]
    stats = {cat: {"FP": 0, "FN": 0, "TP": 0} for cat in categories}
    all_videos = set(gt_map.keys()).union(set(res_map.keys()))
    discrepancies = []

    for video in all_videos:
        actual = gt_map.get(video, "OTHERS")
        predicted = res_map.get(video, "OTHERS")

        # Se sono uguali (entrambi OTHERS o entrambi PERSON, ecc.), è un successo
        if actual == predicted:
            # Contiamo il successo solo per le categorie "interessanti"
            if actual in stats:
                stats[actual]["TP"] += 1
            continue  # Passiamo al prossimo video, niente errori qui

        # Se arriviamo qui, c'è una discrepanza
        
        # Caso 1: L'AI ha visto qualcosa (es. PERSON) in un video che era OTHERS
        if actual == "OTHERS" and predicted in stats:
            stats[predicted]["FP"] += 1
            discrepancies.append((video, actual, predicted))
            
        # Caso 2: L'AI non ha visto nulla (OTHERS) in un video che era importante (es. ANIMAL)
        elif predicted == "OTHERS" and actual in stats:
            stats[actual]["FN"] += 1
            discrepancies.append((video, actual, predicted))
            
        # Caso 3: Scambio di identità (es. GT dice PERSON, AI dice VEHICLE)
        elif actual in stats and predicted in stats:
            stats[actual]["FN"] += 1
            stats[predicted]["FP"] += 1
            discrepancies.append((video, actual, predicted))

    # --- OUTPUT TABELLA ---
    # --- CALCOLO E OUTPUT METRICHE PROFESSIONALI ---
    # Questa tabella sostituisce quella vecchia perché contiene già TP, FP e FN
    header = f"{'CATEGORIA':<12} | {'TP':<5} | {'FP':<5} | {'FN':<5} | {'PRECISION':<10} | {'RECALL':<10}"
    _print(header)
    _print("-" * 70)
    
    cat_accuracies = []
    
    for cat in categories:
        s = stats[cat]
        # Precision: Quanti dei rilevati sono corretti?
        precision = (s["TP"] / (s["TP"] + s["FP"]) * 100) if (s["TP"] + s["FP"]) > 0 else 0
        # Recall: Quanti dei presenti sono stati rilevati?
        recall = (s["TP"] / (s["TP"] + s["FN"]) * 100) if (s["TP"] + s["FN"]) > 0 else 0
        
        _print(f"{cat:<12} | {s['TP']:<5} | {s['FP']:<5} | {s['FN']:<5} | {precision:>9.1f}% | {recall:>8.1f}%")
        
        # Non contiamo OTHERS nella media delle performance AI
        if cat != "OTHERS":
            cat_accuracies.append(recall)
       # --- RIEPILOGO FINALE ---
    avg_recall = sum(cat_accuracies) / len(cat_accuracies) if cat_accuracies else 0
    total_videos = len(all_videos)
    total_tp = sum(s["TP"] for s in stats.values())
    global_acc = (total_tp / total_videos * 100) if total_videos > 0 else 0

    _print("-" * 70)
    _print(f"{'ACCURATEZZA GLOBALE (con Others):':<45} {global_acc:>6.2f}%")
    _print(f"{'RECALL MEDIO CATEGORIE REALI:':<45} {avg_recall:>6.2f}%")
    _print("-" * 70)
    # --- DETTAGLIO ERRORI (Messo qui per non interrompere le tabelle) ---
    if discrepancies:
        _print("🔍 DETTAGLIO ERRORI (Discrepanze):")
        for video, actual, predicted in sorted(discrepancies):
            _print(f"Video: {video:<40} | GT: {actual:<10} | PREDICT: {predicted:<10}")

 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compara GT e Predizioni AI.")
    parser.add_argument("--dir", type=str, help="Directory sessione (cerca nomi file standard)")
    parser.add_argument("--gt", type=str, help="Percorso esplicito al file ground_truth.json")
    parser.add_argument("--res", type=str, help="Percorso esplicito al file classification_results.json")
    args = parser.parse_args()
    
    compare_results(session_dir=args.dir, gt_file=args.gt, res_file=args.res)