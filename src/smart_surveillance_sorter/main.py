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

from colorama import Fore, Style
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="AI Video Sorter")

    # 1. Posizionale: Mode
    parser.add_argument(
    "--mode", 
    choices=["full", "person", "person_animal"], 
    default="full", 
    help="Modalità di analisi: full (tutto), person (solo umani), person_animal (umani e animali)"
)

    # 2. Stringhe e Opzioni
    parser.add_argument("--device", default=None)
    parser.add_argument("--dir", required=True, help="Directory dei video")
    parser.add_argument("--output-dir", dest="output_dir", help="Cartella di destinazione")

    # 3. Booleani per Scanner (dest= corrisponde esattamente all'__init__)
    parser.add_argument("--refine", dest="is_refine", action="store_true")
    parser.add_argument("--fallback", dest="is_fallback", action="store_true")
    parser.add_argument("--test", dest="is_test", action="store_true")
    parser.add_argument("--check-clean", dest="is_check_clean", action="store_true")
    parser.add_argument("--real-time", dest="is_real_time", action="store_true")

    # 4. Motore
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--vision", action="store_const", dest="engine", const="vision")
    group.add_argument("--blip", action="store_const", dest="engine", const="blip")
    parser.set_defaults(engine="blip")

    # 5. Utility (Mancavano queste!)
    parser.add_argument("--ground", action="store_true", help="Genera ground truth")
    parser.add_argument("--compare", action="store_true", help="Compara risultati")
    parser.add_argument("--gt", type=str, help="Path ground_truth.json")
    parser.add_argument("--res", type=str, help="Path classification_results.json")

    return parser.parse_args()



def main():
   
    args = parse_args()
    input_dir = args.dir

  
    log = get_logger(debug=args.is_test)
    if not check_dir(Path(input_dir),is_readable=True):
        log.critical(f"❌ ERRORE CRITICO: cartella di input non esistente")
        sys.exit(1)
    
    if args.output_dir is None:
        output_dir = args.dir
    
    if not check_dir(Path(output_dir),is_writeable=True):
        log.critical(f"❌ ERRORE CRITICO: cartella di output non scrivibile")
        sys.exit(1)


    if args.ground:
        log.info(f"🛠️  Modalità Ground Truth attivata per: {input_dir}")
        try:
            # 1. Genera i dati
            risultati = genera_ground_truth(input_dir, log)
            
            # 2. Definisci il percorso (usando Path per compatibilità con la tua utils)
            output_file_path = Path(output_dir) / "ground_truth.json"
            
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
        gt_path = args.gt if args.gt else Path(output_dir) / "ground_truth.json"
        res_path = args.res if args.res else Path(output_dir) / "classification_results.json"
        
        try:
            # Chiamiamo la funzione di confronto (assicurati di averla importata)
            # Passiamo None a session_dir perché stiamo già risolvendo i percorsi qui
            compare_results(gt_file=gt_path, res_file=res_path, log=log)
            sys.exit(0)
        except Exception as e:
            log.error(f"❌ Errore durante il confronto: {e}")
            sys.exit(1)
    

    log.info(f"Request to scan folder={input_dir} | Mode={args.mode} | Refine={args.is_refine} | Fallback={args.is_fallback}")
    # --- 2. IL TRUCCO DELLO SPLAT (**) ---
    
    # vars(args) crea un dizionario con TUTTI gli argomenti del parser
   # --- LOGICA SCANNER ---
    log.info(f"🚀 Avvio Scanner | Mode={args.mode} | Engine={args.engine}")
    
    # Prepariamo i parametri per lo splat
    params = vars(args).copy()

    # Pulizia: togliamo tutto ciò che NON è nell'__init__ dello Scanner
    to_remove = ['dir', 'output_dir', 'ground', 'compare', 'gt', 'res']
    for key in to_remove:
        params.pop(key, None)

    start_time = time.time()
    try:
        # TRUCCO MAGICO: Passa tutto all'init dello Scanner
        # Python mapperà params['is_refine'] su is_refine=... dell'init
        scanner = Scanner(**params) 
        
        log.info(f"📂 Inizio analisi cartella={input_dir}")
        scanner.scan_folder(input_dir, output_dir)
        
    except KeyboardInterrupt:
        log.warning(f"⚠️ Analisi interrotta dall'utente.")
    except Exception as e:
        log.error(f"💥 Errore critico: {e}")
    finally:
        if 'scanner' in locals():
            scanner._print_final_summary(time.time() - start_time)
    # start_time = time.time()
    # # Otteniamo lo scanner configurato
    # scanner = Scanner(
    #     mode=args.mode,
    #     device=args.device,
    #     is_refine=args.refine,
    #     is_fallback=args.fallback,
    #     is_test=args.test,
    #     engine = args.engine,
    #     is_check_clean=args.check_clean,
    #     is_real_time=args.real_time
    # )
    
    # log.info(f"Scan folder={args.dir} | Mode={args.mode}")

    # # 3. Avvio scansione
    # log.info(f"📂 Inizio analisi cartella={input_dir}")
    # try:
    #     scanner.scan_folder(input_dir, outputDir)
    #     # Usiamo Fore.GREEN per il successo
    #     log.info(f"✅ Analisi completata con successo.")
        
    # except KeyboardInterrupt:
    #     log.warning(f"⚠️ Analisi interrotta dall'utente.")
    # except Exception as e:
    #     log.error(f"💥 Errore critico durante la scansione: {e}")
    #     sys.exit(1)
    
    # end_time = time.time()
    # scanner._print_final_summary(end_time - start_time)

if __name__ == "__main__":
    main()