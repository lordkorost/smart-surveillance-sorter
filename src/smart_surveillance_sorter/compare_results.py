import argparse
import os
from pathlib import Path

from smart_surveillance_sorter.constants import FINAL_REPORT, GROUND_TRUTH
from smart_surveillance_sorter.utils import load_json

def compare_results(session_dir=None, gt_file=None, res_file=None, log=None):
    """
    Compares Ground Truth results with AI predictions.
    
    Calculates per-category metrics such as True Positives, False Positives, 
    and recall (excluding 'OTHERS'), printing a summary table and discrepancy details.
    
    Args:
        session_dir (str): Input directory containing the data.
        gt_file (str): Path to the ground_truth.json file.
        res_file (str): Path to the classification_results.json file.
        log (logging.Logger): Optional logging handle for output messages.

    Returns:
        None: Prints global accuracy and average recall directly.
    """
    def _print(msg): log.info(msg) if log else print(msg)
    def _err(msg): log.error(msg) if log else print(f"❌ {msg}")

    # --- Path resolve logic ---
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
        _err("Use a valid --dir or direct path with --gt and --res")
        return

    gt_list  = load_json(path_gt)
    res_data = load_json(path_res)

    if gt_list is None or res_data is None:
        _err("Error loading json: One or all file missing or corrupted.")
        return

    # GT map
    gt_map = {os.path.basename(item["video_name"]): item["category"].upper() for item in gt_list}

    # AI res
    if isinstance(res_data, list):
        res_map = {os.path.basename(item["video_name"]): item["category"].upper() for item in res_data}
    else:
        res_map = {os.path.basename(path): data["video_category"].upper() for path, data in res_data.items()}

    # --- Intersection all videos and missing videos ---
    all_videos  = set(gt_map.keys()).intersection(set(res_map.keys()))
    only_in_gt  = set(gt_map.keys()) - set(res_map.keys())
    only_in_res = set(res_map.keys()) - set(gt_map.keys())

    # --- Calculate stats ---
    categories = ["PERSON", "VEHICLE", "ANIMAL", "OTHERS"]
    stats = {cat: {"FP": 0, "FN": 0, "TP": 0} for cat in categories}
    discrepancies = []

    for video in all_videos:
        actual    = gt_map.get(video, "OTHERS")
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

    
    total_videos      = len(all_videos)
    total_tp          = sum(s["TP"] for s in stats.values())
    total_misclassified = len(discrepancies)
    global_acc        = (total_tp / total_videos * 100) if total_videos > 0 else 0

    _print("=" * 65)
    _print(f"  Total videos compared : {total_videos}")
    if only_in_gt:
        _print(f"In GT but not in results : {len(only_in_gt)} (not processed?)")
    if only_in_res:
        _print(f"In results but not in GT : {len(only_in_res)}")
    _print(f"  Correctly classified  : {total_tp:<5} ({global_acc:.1f}%)")
    _print(f"  Misclassified         : {total_misclassified:<5} ({100 - global_acc:.1f}%)")
    _print("=" * 65)

    # --- tab for category ---
    _print(f"{'CATEGORY':<12} | {'TP':<5} | {'FP':<5} | {'FN':<5} | {'PRECISION':<10} | {'RECALL':<8}")
    _print("-" * 65)

    cat_recalls = []
    for cat in categories:
        s = stats[cat]
        precision = (s["TP"] / (s["TP"] + s["FP"]) * 100) if (s["TP"] + s["FP"]) > 0 else 0
        recall    = (s["TP"] / (s["TP"] + s["FN"]) * 100) if (s["TP"] + s["FN"]) > 0 else 0
        _print(f"{cat:<12} | {s['TP']:<5} | {s['FP']:<5} | {s['FN']:<5} | {precision:>9.1f}% | {recall:>7.1f}%")
        if cat != "OTHERS":
            cat_recalls.append(recall)

    avg_recall = sum(cat_recalls) / len(cat_recalls) if cat_recalls else 0
    _print("-" * 65)
    _print(f"  Global accuracy            : {global_acc:.2f}%")
    _print(f"  Avg recall (excl. Others)  : {avg_recall:.2f}%")
    _print("=" * 65)

    # --- Error detail ---
    if discrepancies:
        _print(f"\n  Misclassified videos ({len(discrepancies)}):")
        _print(f"  {'VIDEO':<45} | {'GT':<10} | {'PREDICTED'}")
        _print("  " + "-" * 70)
        for video, actual, predicted in sorted(discrepancies):
            _print(f"  {video:<45} | {actual:<10} | {predicted}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare GT with AI.")
    parser.add_argument("--dir", type=str, help="Input directory")
    parser.add_argument("--gt",  type=str, help="Path to ground_truth.json")
    parser.add_argument("--res", type=str, help="Path to classification_results.json")
    args = parser.parse_args()
    compare_results(session_dir=args.dir, gt_file=args.gt, res_file=args.res)