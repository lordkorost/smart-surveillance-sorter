from pathlib import Path
import logging
import sys
from pathlib import Path

# Aggiunge la cartella 'src' al percorso di ricerca di Python
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

# Ora l'import dovrebbe funzionare usando il nome del pacchetto
from smart_surveillance_sorter.scanners.base import ScannerBase
from smart_surveillance_sorter.scanners.base import ScannerBase

# Configurazione minima per il test
dummy_settings = {
    "config_files": {
        "cameras_path": "config/cameras.json",
        "prompts_path": "config/prompts.json",
        "models_dir": "models"
    },
    "yolo_settings": {
        "model_path": "yolov8l.pt",
        "threshold": 0.25
    }
}

def test_run():
    logging.basicConfig(level=logging.INFO)
    
    # Inizializziamo lo scanner
    # (Assicurati che config/cameras.json e prompts.json esistano!)
    scanner = ScannerBase(
        settings=dummy_settings,
        mode="person",
        is_refine=False,
        is_fallback=False
    )

    # Cartella da scansionare (cambiala con una tua cartella reale con qualche file Reolink)
    input_dir = Path("./fails") 
    
    if not input_dir.exists():
        print(f"Creare la cartella {input_dir} e metterci qualche file .mp4 e .jpg per il test!")
        return

    print("\n--- AVVIO TEST PIPELINE ---")
    scanner.scan_folder(input_dir)
    
    print("\n--- RISULTATI IN MEMORIA ---")
    for res in scanner.results:
        print(f"Video: {Path(res['video_path']).name}")
        print(f"  Trovato: {res['categories_found']}")
        print(f"  Risolto da: {res.get('resolved_by')}")
        print(f"  Frame path: {res['frames'][0]['path']}")
    
    print(f"\nVideo totali risolti: {len(scanner.resolved_videos)}")

if __name__ == "__main__":
    test_run()