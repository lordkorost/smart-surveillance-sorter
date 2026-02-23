import os
import json
from pathlib import Path
import sys
import argparse
from collections import defaultdict

from smart_surveillance_sorter.logger import get_logger
from smart_surveillance_sorter.utils import check_dir, save_json

def genera_ground_truth(root_dir, log):
    ground_truth = []
    valid_categories = {'person', 'animal', 'others', 'nothing', 'vehicle'}
    
    for root, dirs, files in os.walk(root_dir):
        category = os.path.basename(root).lower()
        
        if category in valid_categories:
            for file in files:
                if file.endswith(('.mp4', '.mkv', '.avi')):
                    entry = {
                        "video_name": file,
                        "category": category
                    }
                    ground_truth.append(entry)
                    log.debug(f"Map: {file} -> {category}")
                    
    return ground_truth

def check_duplicates_with_log(root_dir, log):
    file_map = defaultdict(list)
    valid_categories = {'person', 'animal', 'others', 'nothing', 'vehicle'}

    for root, dirs, files in os.walk(root_dir):
        category = os.path.basename(root).lower()
        if category in valid_categories:
            for file in files:
                if file.endswith(('.mp4', '.mkv', '.avi')):
                    file_map[file].append(category)

    duplicates = {name: cats for name, cats in file_map.items() if len(cats) > 1}

    if duplicates:
        log.warning(f"Found num_vid={len(duplicates)} duplicate")
        for name, cats in duplicates.items():
            log.warning(f"  - {name} is in: {', '.join(cats)}")
    else:
        log.info("No duplicate videos found. Dataset is perfect.")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Create Ground Truth from manual sorted videos for testing script accuracy.")
    parser.add_argument("--dir", type=str, required=True, help="Input directory of sorted videos")
    parser.add_argument("--outputDir", type=str, default=None, help="Output directory to save json. (default: input dir)")
    parser.add_argument("--test", action="store_true", help="Activate debug log")
    args = parser.parse_args()

    # 1. Inizializza il logger professionale anche qui
    log = get_logger(debug=args.test)
    
    input_dir = args.dir
    output_dir = args.outputDir if args.outputDir else input_dir

    # 2. Controllo sicurezza Input
    if not check_dir(Path(input_dir), is_readable=True):
        log.critical(f"Folder={input_dir} not exists or is not redeable.")
        sys.exit(1)
    
    # 3. Controllo sicurezza Output
    if not check_dir(Path(output_dir), is_writeable=True):
        log.critical(f"Folder={output_dir} is not writeable.")
        sys.exit(1)

    # 4. Generazione Dati
    log.info(f"Start ground truth generate on folder={input_dir}")
    risultati = genera_ground_truth(input_dir, log)
    
    # 5. Salvataggio con la tua utility
    output_file_path = Path(output_dir) / "ground_truth.json"
    if save_json(risultati, output_file_path):
        log.info(f"✅ Ground Truth create. File={output_file_path}")
        log.debug(f"Num_vid={len(risultati)}")
        
        # 6. Verifica duplicati
        check_duplicates_with_log(input_dir, log)
    else:
        # Il log.critical è già dentro save_json, quindi qui basta uscire
        sys.exit(1)