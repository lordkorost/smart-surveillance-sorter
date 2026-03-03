import argparse
import os
import shutil
from pathlib import Path

from smart_surveillance_sorter.constants import FINAL_REPORT, GROUND_TRUTH
from smart_surveillance_sorter.utils import load_json

def compare_results(session_dir=None, gt_file=None, res_file=None, copy_wrong=None, log=None):
    def _print(msg): log.info(msg) if log else print(msg)
    def _err(msg): log.error(msg) if log else print(f"❌ {msg}")

    # --- LOGICA DI RISOLUZIONE PERCORSI ---
    path_gt = None
    path_res = None

    if session_dir:
        s_path = Path(session_dir)
        path_gt = s_path / GROUND_TRUTH
        path_res = s_path / FINAL_REPORT
    
    if gt_file:
        path_gt = Path(gt_file)
    if res_file:
        path_res = Path(res_file)

    if not path_gt or not path_res or not path_gt.exists() or not path_res.exists():
        _err(f"File not found!\nGT: {path_gt}\nRES: {path_res}")
        _err("Use valid --dir or  path to files with --gt e --res")
        return

    _print(f"📊 Compare...\n📖 GT: {path_gt}\n🤖 AI: {path_res}")

    gt_list = load_json(path_gt)
    res_data = load_json(path_res)

    if gt_list is None or res_data is None:
        _err("Error loading JSON: one or all files are missing or corrupt.")
        return

    gt_map = {os.path.basename(item["video_name"]): item["category"].upper() for item in gt_list}

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

        if actual == predicted:
            if actual in stats:
                stats[actual]["TP"] += 1
            continue

        if actual == "OTHERS" and predicted in stats:
            stats[predicted]["FP"] += 1
            discrepancies.append((video, actual, predicted))
        elif predicted == "OTHERS" and actual in stats:
            stats[actual]["FN"] += 1
            discrepancies.append((video, actual, predicted))
        elif actual in stats and predicted in stats:
            stats[actual]["FN"] += 1
            stats[predicted]["FP"] += 1
            discrepancies.append((video, actual, predicted))

    # --- OUTPUT TABELLA ---
    header = f"{'CATEGORY':<12} | {'TP':<5} | {'FP':<5} | {'FN':<5} | {'PRECISION':<10} | {'RECALL':<10}"
    _print(header)
    _print("-" * 70)
    
    cat_accuracies = []
    
    for cat in categories:
        s = stats[cat]
        precision = (s["TP"] / (s["TP"] + s["FP"]) * 100) if (s["TP"] + s["FP"]) > 0 else 0
        recall = (s["TP"] / (s["TP"] + s["FN"]) * 100) if (s["TP"] + s["FN"]) > 0 else 0
        _print(f"{cat:<12} | {s['TP']:<5} | {s['FP']:<5} | {s['FN']:<5} | {precision:>9.1f}% | {recall:>8.1f}%")
        if cat != "OTHERS":
            cat_accuracies.append(recall)

    avg_recall = sum(cat_accuracies) / len(cat_accuracies) if cat_accuracies else 0
    total_videos = len(all_videos)
    total_tp = sum(s["TP"] for s in stats.values())
    global_acc = (total_tp / total_videos * 100) if total_videos > 0 else 0

    _print("-" * 70)
    _print(f"{'Global accuracy (with Others):':<45} {global_acc:>6.2f}%")
    _print(f"{'RECALL on real category:':<45} {avg_recall:>6.2f}%")
    _print("-" * 70)

    if discrepancies:
        _print("Difference list:")
        for video, actual, predicted in sorted(discrepancies):
            _print(f"Video={video:<40} | GT={actual:<10} | PREDICT={predicted:<10}")

    # --- COPIA VIDEO SBAGLIATI ---
    if copy_wrong and discrepancies:
        source_dir = path_gt.parent
        
        # Sottocartelle per tipo di errore: GT_ANIMAL/PRED_OTHERS etc.
        wrong_root = Path(copy_wrong)
        copied = 0
        not_found = 0

        _print(f"\n📂 Copio {len(discrepancies)} video sbagliati in {wrong_root}...")

        for video_name, actual, predicted in discrepancies:
            # Sottocartella GT_X__PRED_Y per organizzare per tipo di errore
            sub_dir = wrong_root / f"GT_{actual}__PRED_{predicted}"
            sub_dir.mkdir(parents=True, exist_ok=True)

            candidates = list(source_dir.rglob(video_name))
            if candidates:
                src = candidates[0]
                dst = sub_dir / video_name  # filename originale intatto
                try:
                    shutil.copy2(src, dst)
                    copied += 1
                except Exception as e:
                    _print(f"  ⚠️ Error copy {video_name}: {e}")
            else:
                _print(f"  ⚠️ Video not found: {video_name}")
                not_found += 1

        _print(f"  ✅ Copied {copied}/{len(discrepancies)} video")
        if not_found:
            _print(f"  ⚠️ Not found: {not_found}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare GT with AI.")
    parser.add_argument("--dir", type=str, help="Input directory, search ground_truth.json and classification_results.json")
    parser.add_argument("--gt", type=str, help="Path to file ground_truth.json")
    parser.add_argument("--res", type=str, help="Path to file classification_results.json")
    parser.add_argument("--copy-wrong", type=str, metavar="DEST_DIR",
                        help="Copy wrong videos in folder for error type")
    args = parser.parse_args()
    
    compare_results(session_dir=args.dir, gt_file=args.gt, res_file=args.res, copy_wrong=args.copy_wrong)