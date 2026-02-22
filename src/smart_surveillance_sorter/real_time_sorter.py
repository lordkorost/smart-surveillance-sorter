import os
import time
import subprocess
import argparse
import logging
from pathlib import Path

# Configurazione log base
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def run_loop():
    parser = argparse.ArgumentParser(description="Real-time NVR Surveillance Sorter")
    parser.add_argument("--dir", required=True, help="Cartella input NVR")
    parser.add_argument("--mode", default="person", help="Categoria da cercare")
    parser.add_argument("--output-dir", help="Cartella output (opzionale)")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--interval", type=int, default=60, help="Secondi tra scansioni")
    parser.add_argument("--test", action="store_true", help="Abilita log dettagliati (DEBUG) e test mode")
   
    parser.add_argument("--refine", action="store_true", help="Attiva refine nello scanner")
    parser.add_argument("--blip", action="store_true", help="Usa motore BLIP (default)")
    parser.add_argument("--vision", action="store_true", help="Usa motore Vision API")
    parser.add_argument("--fallback", action="store_true")
    args = parser.parse_args()

    # Se --test è attivo, alziamo il livello di log del Sorter stesso
    if args.test:
        log.setLevel(logging.DEBUG)
        log.debug("🧪 Modalità TEST/DEBUG attiva: i log saranno molto dettagliati.")

    log.info(f"🚀 Monitorando: {args.dir} ogni {args.interval}s")

    while True:
        try:
            # In questa (se main.py è nella stessa cartella del sorter):
            import sys
            from pathlib import Path

            # Ottieni il path assoluto della cartella dove si trova questo script (sorter)
            script_dir = Path(__file__).parent.absolute()
            main_path = script_dir / "main.py"


            # Costruiamo il comando base
            cmd = [
                "python", str(main_path),
                "--dir", args.dir,
                "--mode", args.mode,
            ]

            # Aggiungiamo gli optional solo se presenti
            if args.device:
                cmd.extend(["--device", args.device])

            if args.output_dir:
                cmd.extend(["--output-dir", args.output_dir])

            if args.test:
                cmd.append("--test")

            if args.refine: cmd.append("--refine")
           # Gestione motore (priorità a Vision se presente, altrimenti Blip)
            if args.vision:
                cmd.append("--vision")
            else:
                cmd.append("--blip")
                if args.fallback:
                    cmd.append("--fallback")

            log.info(f"--- Inizio scansione ({args.mode}) ---")
            
            # Eseguiamo lo scanner. 
            # NOTA: subprocess.run eredita automaticamente stdout/stderr, 
            # quindi vedrai i log di main.py direttamente nel terminale del sorter.
            result = subprocess.run(cmd)
            
            if result.returncode == 0:
                log.info(f"✅ Giro completato. Prossimo check tra {args.interval}s")
            else:
                log.warning(f"⚠️ Lo scanner ha restituito un errore (Exit Code: {result.returncode})")

        except KeyboardInterrupt:
            log.info("👋 Arresto richiesto dall'utente.")
            break
        except Exception as e:
            log.error(f"💥 Errore imprevisto nel loop: {e}")

        time.sleep(args.interval)

if __name__ == "__main__":
    run_loop()