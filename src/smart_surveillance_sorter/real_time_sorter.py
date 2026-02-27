import time
import argparse
import logging
from pathlib import Path

import torch
from smart_surveillance_sorter.logger import get_logger
from smart_surveillance_sorter.scanners.scanner import Scanner
from smart_surveillance_sorter.utils import cleanup



def run_loop():
    parser = argparse.ArgumentParser(description="Real-time NVR Surveillance Sorter")
    parser.add_argument("--dir",        required=True)
    parser.add_argument("--output-dir", dest="output_dir")
    parser.add_argument("--mode",       default="person")
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--interval",   type=int, default=60)
    parser.add_argument("--refine",     action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--vision", action="store_const", dest="engine", const="vision")
    group.add_argument("--blip",   action="store_const", dest="engine", const="blip")
    parser.set_defaults(engine="blip")
    args = parser.parse_args()

    output_dir = args.output_dir or args.dir
    log = get_logger(debug=False)
    log.info(f"Watching folder={args.dir} every {args.interval}s")

    while True:
        try:
            log.info(f"--- Inizio scansione ({args.mode}) ---")
            scanner = Scanner(
                mode=args.mode,
                device=args.device,
                is_refine=args.refine,
                is_fallback=False,
                is_test=False,
                engine=args.engine,
                is_check_clean=False,
            )
            scanner.scan_folder(args.dir, output_dir)
            log.info(f"Ciclo completato. Prossimo in {args.interval}s")
        except (KeyboardInterrupt, RuntimeError) as e:
            log.info(f"Stop: {e}")
            break
        except Exception as e:
            log.error(f"Errore nel ciclo: {e}")
        finally:
            # Pulizia memoria dopo ogni ciclo
            del scanner
            cleanup()
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass

        time.sleep(args.interval)

if __name__ == "__main__":
    run_loop()