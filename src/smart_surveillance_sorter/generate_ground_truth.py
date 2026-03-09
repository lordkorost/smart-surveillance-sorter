import os
from pathlib import Path
import sys
import argparse
from collections import defaultdict

from smart_surveillance_sorter.logger import get_logger
from smart_surveillance_sorter.utils import check_dir, save_json

def genera_ground_truth(root_dir, log):
    """Generates ground truth data by scanning a directory structure and mapping video files to categories.

    This function iterates through the provided root directory, identifying folders that match valid 
    category names (e.g., 'person', 'animal', 'others', 'nothing', 'vehicle'). It collects video files 
    ending with '.mp4', '.mkv', or '.avi' within these folders and maps them to their respective categories.
    
    Args:
        root_dir (str | Path): The path to the input directory containing sorted videos.
        log (logging.Logger): Optional logger for debugging output messages.
        
    Returns:
        list[dict]: A list of dictionaries where each dict contains 'video_name' and 'category'.
    
    Note:
        Categories must match the predefined valid set defined in the function logic.
    """
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
    """Checks for duplicate files in the input directory and logs them.
    
    Prevents redundant processing or overwrites by verifying unique file paths 
    before generation starts. This step ensures data integrity across the session [1].
    
    Args:
        input_dir (Path | str): Path to the input folder being checked for duplicates.
        log (logging.Logger): Logger handle to record warnings or errors found [1].
    
    Returns:
        None: Performs in-place logging checks without returning a specific value [1].
    """
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
    """Entry point for Ground Truth generation from sorted videos.
    
    This script scans the input directory to build a ground truth mapping based on 
    folder categories (e.g., 'person', 'vehicle'). It validates read/write permissions,
    generates the ground_truth.json file, logs statistics, and checks for duplicate files.

    Args:
        args (argparse.Namespace): Parsed arguments containing:
            - dir: Path to input directory of sorted videos (required).
            - outputDir: Path to output directory (default: input_dir).
            - test: Flag to activate debug logging mode.
        
    Returns:
        None: Exits the program with status code 0 on success or non-zero on failure [1].

    Side Effects:
        Creates ground_truth.json in the specified output directory.
        Logs warnings for duplicates, access errors, or writeable issues.
        Calls sys.exit(1) if critical checks fail (missing folders, permissions).

    Example:
        python generate_ground_truth.py --dir /path/to/videos --test
    """
    parser = argparse.ArgumentParser(description="Create Ground Truth from manual sorted videos for testing script accuracy.")
    parser.add_argument("--dir", type=str, required=True, help="Input directory of sorted videos")
    parser.add_argument("--outputDir", type=str, default=None, help="Output directory to save json. (default: input dir)")
    parser.add_argument("--test", action="store_true", help="Activate debug log")
    args = parser.parse_args()

   
    log = get_logger(debug=args.test)
    
    input_dir = args.dir
    output_dir = args.outputDir if args.outputDir else input_dir

    # 2. Input security check
    if not check_dir(Path(input_dir), is_readable=True):
        log.critical(f"Folder={input_dir} not exists or is not redeable.")
        sys.exit(1)
    
    # 3. Output security check
    if not check_dir(Path(output_dir), is_writeable=True):
        log.critical(f"Folder={output_dir} is not writeable.")
        sys.exit(1)

    # 4. Generate
    log.info(f"Start ground truth generate on folder={input_dir}")
    risultati = genera_ground_truth(input_dir, log)
    
    # 5. Save
    output_file_path = Path(output_dir) / "ground_truth.json"
    if save_json(risultati, output_file_path):
        log.info(f"Ground Truth create. File={output_file_path}")
        log.debug(f"Num_vid={len(risultati)}")
        
        # 6. Check duplicates
        check_duplicates_with_log(input_dir, log)
    else:
        sys.exit(1)