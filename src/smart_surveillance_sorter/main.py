import argparse
import logging
import sys
from pathlib import Path
import time

from smart_surveillance_sorter.constants import PROJECT_ROOT
from smart_surveillance_sorter.log_config import configure_logger
from smart_surveillance_sorter.scanners.scanner import Scanner
from smart_surveillance_sorter.utils import check_dir

from colorama import Fore, Style, init

# # Inizializza colorama per i colori ANSI
# init(autoreset=True)

# # Formattatore personalizzato: TIME in Verde, il resto normale
# class SurveillanceFormatter(logging.Formatter):
#     def format(self, record):
#         # [HH:MM:SS] in verde
#         timestamp = f"{Fore.GREEN}{self.formatTime(record, '%H:%M:%S')}{Style.RESET_ALL}"
#         # Messaggio log
#         return f"[{timestamp}] - {record.getMessage()}"

# # Decide level once
# DEBUG = os.getenv("DEBUG", "0") == "1"

#configure_logger(debug=DEBUG, log_file="app.log")  # <‑‑ configure once
#log = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Smart Surveillance Sorter - AI Video Classifier")
    parser.add_argument("--dir", type=str, required=True, help="Directory contenente i video da analizzare")
    parser.add_argument("--mode", type=str, choices=["full", "person", "person_animal"], default="full", 
                        help="Modalità di analisi (default: full)")
    parser.add_argument("--device", type=str, default=None, 
                    help="Device su cui far girare YOLO (es: 'cpu', '0', 'cuda', 'mps')")
    
    #parser.add_argument("--dir", type=str, required=True, help="Directory contenente i video da analizzare")
    parser.add_argument("--outputDir",type=str, default=None, help="Output Directory")
    
    parser.add_argument("--refine", action="store_true", help="Usa Vision per confermare i rilevamenti YOLO")
    parser.add_argument("--fallback", action="store_true", help="Analisi finale su immagini NVR per video 'nothing'")
    parser.add_argument(
        "--test", 
        action="store_true", 
        help="Modalità test: copia i file invece di spostarli e aumenta i log di debug."
    )
    # Scelta del motore (escludenti a vicenda per pulizia)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--vision", action="store_const", dest="engine", const="vision", help="Usa Vision AI (Ollama/Qwen)")
    group.add_argument("--blip", action="store_const", dest="engine", const="blip", help="Usa BLIP")

    # Aggiungiamo il flag per il controllo pulizia
    parser.add_argument("--check-clean", action="store_true", 
                        help="Esegue il controllo salute lenti (Lens Health Check) usando le immagini in /checks")
    
    # Default se --refine è attivo ma non specifichi il motore
    parser.set_defaults(engine="blip")
    
    args = parser.parse_args()
    input_dir = args.dir

    configure_logger(debug=args.test)  # <‑‑ configure once
    log = logging.getLogger(__name__)

    if not check_dir(Path(input_dir),is_readable=True):
        log.critical(f"❌ ERRORE CRITICO: cartella di input non esistente")
        sys.exit(1)
    
    if args.outputDir is None:
        outputDir = args.dir
    
    if not check_dir(Path(outputDir),is_writeable=True):
        log.critical(f"❌ ERRORE CRITICO: cartella di output non scrivibile")
        sys.exit(1)
    


    #print(f"🚀 Root del progetto: {PROJECT_ROOT}")
    log.info(f"Request to scan {input_dir} with Mode:{args.mode},Refine Pass:{args.refine},Fallback Pass:{args.fallback}")
    start_time = time.time()
    # Otteniamo lo scanner configurato
    scanner = Scanner(
        mode=args.mode,
        device=args.device,
        is_refine=args.refine,
        is_fallback=args.fallback,
        is_test=args.test,
        engine = args.engine,
        is_check_clean=args.check_clean
    )
    
    log.info(f"Scan folder: {Fore.YELLOW}{args.dir}{Style.RESET_ALL} | Mode: {Fore.CYAN}{args.mode}{Style.RESET_ALL}")

    # 3. Avvio scansione
    log.info(f"📂 Inizio analisi cartella: {Fore.YELLOW}{input_dir}{Style.RESET_ALL}")
    try:
        scanner.scan_folder(input_dir, outputDir)
        # Usiamo Fore.GREEN per il successo
        log.info(f"{Fore.GREEN}✅ Analisi completata con successo.{Style.RESET_ALL}")
        
    except KeyboardInterrupt:
        log.warning(f"{Fore.YELLOW}⚠️ Analisi interrotta dall'utente.{Style.RESET_ALL}")
    except Exception as e:
        log.error(f"{Fore.RED}💥 Errore critico durante la scansione: {e}{Style.RESET_ALL}")
        sys.exit(1)
    
    end_time = time.time()
    scanner._print_final_summary(end_time - start_time)

if __name__ == "__main__":
    main()