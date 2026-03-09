import argparse
import sys
from pathlib import Path
import time
from smart_surveillance_sorter.compare_results import compare_results
from smart_surveillance_sorter.constants import FINAL_REPORT, GROUND_TRUTH, LENS_HEALTH
from smart_surveillance_sorter.file_utils import associate_files, build_index
from smart_surveillance_sorter.generate_ground_truth import check_duplicates_with_log, genera_ground_truth
from smart_surveillance_sorter.logger import get_logger
from smart_surveillance_sorter.scanners.scanner import Scanner
from smart_surveillance_sorter.utils import check_dir, cleanup, save_json

import argparse

def parse_args():
    """Parse command-line arguments for the surveillance sorter.
    
    Returns:
        Namespace object with parsed arguments
    """
    parser = argparse.ArgumentParser(description="AI Surveillance Video Sorter")

    # 1. Posizionale: Mode
    parser.add_argument(
    "-m","--mode", 
    choices=["full", "person", "person_animal"], 
    default="full", 
    help="Processing mode: 'person' to sort only video with person, 'person-animal' to sort video with person and animal, 'full' to sort video with person,animal and vehicle"
)

    # 2. Stringhe e Opzioni
    parser.add_argument("--device", default=None)
    parser.add_argument("-d","--dir", required=True, help="Directory containing the videos to scan.")
    parser.add_argument("-o","--output-dir", dest="output_dir", help="Path where the sorted video will be moved")

    # 3. Booleani per Scanner (dest=  a __init__)
    parser.add_argument("--refine", dest="is_refine", action="store_true", help="Activate refine step after yolo, need --vision or --blip")
    parser.add_argument("--fallback", dest="is_fallback", action="store_true", help="Activate fallback vision step after yolo and blip")
    parser.add_argument("--test", dest="is_test", action="store_true", help="Debug log,copy sorted video instead move, metric log")
    parser.add_argument("--check-clean", dest="is_check_clean", action="store_true",help="Activate vision cameras check clean step")
    parser.add_argument("--no-sort", dest="is_sort",action="store_false",help="False to deactivate moving/copyng file")

    # 4. Motore
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--vision", action="store_const", dest="engine", const="vision", help="Use ollama and vl models after yolo")
    group.add_argument("--blip", action="store_const", dest="engine", const="blip", help="Use blip and clip after yolo step")
    parser.set_defaults(engine="blip")

    # 5. Utility
    parser.add_argument("--ground", action="store_true", help="Generate ground truth for the input dir")
    parser.add_argument("--compare", action="store_true", help="Compare results between ground truth and results of the sorter json")
    parser.add_argument("--gt", type=str, help="Path ground_truth.json")
    parser.add_argument("--res", type=str, help="Path classification_results.json")

    return parser.parse_args()



def main():
    """Main entry point for the surveillance video sorter application.
    
    Orchestrates the complete workflow: argument parsing, validation, scanning,
    and optional ground truth generation or result comparison.
    """
    args = parse_args()
    input_dir = args.dir

  
    log = get_logger(debug=args.is_test)
    if not check_dir(Path(input_dir),is_readable=True):
        log.critical(f"Input dir not exists or is not readable!")
        sys.exit(1)
    
    if args.output_dir is None:
        output_dir = args.dir
    else:
        output_dir = args.output_dir
    if not check_dir(Path(output_dir),is_writeable=True):
        log.critical(f"Output dir is not writeable!")
        sys.exit(1)


    if args.ground:
        log.info(f"Generate Ground Truth for dir={input_dir}")
        try:
            risultati = genera_ground_truth(input_dir, log)
            
            output_file_path = Path(output_dir) / GROUND_TRUTH

            if save_json(risultati, output_file_path):
                log.info(f"Ground Truth generate with success file={output_file_path}")
                log.info(f"Video found {len(risultati)}")

                check_duplicates_with_log(input_dir, log)
            else:
                log.error("Fail to save file=ground_truth.json")
                sys.exit(1)
                
            sys.exit(0)
            
        except Exception as e:
            log.error(f"❌Error during generate Ground Truth: error={e}")
            sys.exit(1)

    # --- COMPARISON MODE ---
    if args.compare:
        log.info(f"Compare mode")
        gt_path = args.gt if args.gt else Path(output_dir) / GROUND_TRUTH
        res_path = args.res if args.res else Path(output_dir) / FINAL_REPORT
        try:
            compare_results(gt_file=gt_path, res_file=res_path, log=log)
            sys.exit(0)
        except Exception as e:
            log.error(f"Error during compare  e={e}")
            sys.exit(1)
    
    log.info(
            f"Scanner start: input-folder={input_dir} | "
            f"output-folder={args.output_dir} | "
            f"Mode={args.mode} | Refine={args.is_refine} | "
            f"Engine={args.engine} | Fallback={args.is_fallback} | "
            f"Test={args.is_test}"
        )
    
    # --- SCANNER LOGIC ---
    # parametri per lo splat
    params = vars(args).copy()

    # Cleanup: remove everything NOT in Scanner.__init__
    to_remove = ['dir', 'output_dir', 'ground', 'compare', 'gt', 'res']
    for key in to_remove:
        params.pop(key, None)

        # --- CHECK CLEAN STANDALONE ---
    if args.is_check_clean and not args.is_refine:
        log.info(f"Check Clean standalone mode — input={input_dir}")
        try:
            scanner = Scanner(**params)
            raw_index = build_index(input_dir, scanner.settings)
            scanner.full_index = associate_files(raw_index, Path(input_dir))
            lens_status = scanner.check_cameras_clean()
            health_report_path = Path(output_dir) / LENS_HEALTH
            save_json(lens_status, health_report_path)
            log.info(f"Report saved in={health_report_path}")
            scanner._print_lens_status()
        except Exception as e:
            log.error(f"Error during check clean: error={e}")
        finally:
            cleanup()
        sys.exit(0)

    start_time = time.time()
    try:
        # Passa tutto all'init dello Scanner
        scanner = Scanner(**params) 
        scanner.scan_folder(input_dir, output_dir)
        
    except KeyboardInterrupt:
        log.warning(f"Scanner interrupted by user.")
    except Exception as e:
        log.error(f"Error={e}")
    finally:
        if 'scanner' in locals():
            scanner._print_final_summary(time.time() - start_time)
        cleanup()


if __name__ == "__main__":
    main()