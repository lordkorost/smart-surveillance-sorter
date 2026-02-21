import argparse
import logging
import sys
from pathlib import Path
import time

from smart_surveillance_sorter.compare_results import compare_results
from smart_surveillance_sorter.constants import PROJECT_ROOT

from smart_surveillance_sorter.generate_ground_truth import check_duplicates_with_log, genera_ground_truth
from smart_surveillance_sorter.logger import get_logger
from smart_surveillance_sorter.scanners.scanner import Scanner
from smart_surveillance_sorter.utils import check_dir, save_json

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
    parser.add_argument("--ground", action="store_true", help="Genera il file ground_truth.json dai video già smistati")
    parser.add_argument("--compare", action="store_true", help="Compara Ground Truth e risultati AI")
    parser.add_argument("--gt", type=str, help="Percorso specifico ground_truth.json (opzionale)")
    parser.add_argument("--res", type=str, help="Percorso specifico classification_results.json (opzionale)")
    args = parser.parse_args()
    input_dir = args.dir

  
    log = get_logger(debug=args.test)
    if not check_dir(Path(input_dir),is_readable=True):
        log.critical(f"❌ ERRORE CRITICO: cartella di input non esistente")
        sys.exit(1)
    
    if args.outputDir is None:
        outputDir = args.dir
    
    if not check_dir(Path(outputDir),is_writeable=True):
        log.critical(f"❌ ERRORE CRITICO: cartella di output non scrivibile")
        sys.exit(1)


    if args.ground:
        log.info(f"🛠️  Modalità Ground Truth attivata per: {input_dir}")
        try:
            # 1. Genera i dati
            risultati = genera_ground_truth(input_dir, log)
            
            # 2. Definisci il percorso (usando Path per compatibilità con la tua utils)
            output_file_path = Path(outputDir) / "ground_truth.json"
            
            # 3. Usa la tua utility di salvataggio
            if save_json(risultati, output_file_path):
                log.info(f"✅ Ground Truth generato con successo: {output_file_path}")
                log.info(f"📊 Totale video mappati: {len(risultati)}")
                
                # 4. Controllo duplicati
                check_duplicates_with_log(input_dir, log)
            else:
                log.error("❌ Fallito il salvataggio del file ground_truth.json")
                sys.exit(1)
                
            sys.exit(0)
            
        except Exception as e:
            log.error(f"❌ Errore durante la generazione del Ground Truth: {e}")
            sys.exit(1)

    # --- MODALITÀ CONFRONTO ---
    if args.compare:
        log.info(f"📊 Modalità Confronto attivata")
        
        # Se non vengono passati --gt o --res, usiamo i default nella outputDir
        gt_path = args.gt if args.gt else Path(outputDir) / "ground_truth.json"
        res_path = args.res if args.res else Path(outputDir) / "classification_results.json"
        
        try:
            # Chiamiamo la funzione di confronto (assicurati di averla importata)
            # Passiamo None a session_dir perché stiamo già risolvendo i percorsi qui
            compare_results(gt_file=gt_path, res_file=res_path, log=log)
            sys.exit(0)
        except Exception as e:
            log.error(f"❌ Errore durante il confronto: {e}")
            sys.exit(1)
    

    log.info(f"Request to scan folder={input_dir} | Mode={args.mode} | Refine={args.refine} | Fallback={args.fallback}")
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
    
    log.info(f"Scan folder={args.dir} | Mode={args.mode}")

    # 3. Avvio scansione
    log.info(f"📂 Inizio analisi cartella={input_dir}")
    try:
        scanner.scan_folder(input_dir, outputDir)
        # Usiamo Fore.GREEN per il successo
        log.info(f"✅ Analisi completata con successo.")
        
    except KeyboardInterrupt:
        log.warning(f"⚠️ Analisi interrotta dall'utente.")
    except Exception as e:
        log.error(f"💥 Errore critico durante la scansione: {e}")
        sys.exit(1)
    
    end_time = time.time()
    scanner._print_final_summary(end_time - start_time)

if __name__ == "__main__":
    main()