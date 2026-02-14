from collections import defaultdict
import os
import json
import sys

def genera_ground_truth(root_dir):
    ground_truth = []
    
    # Categorie che ci aspettiamo di trovare
    valid_categories = {'person', 'animal', 'others', 'vehicle'}

    # Cammina nella struttura: root / camera_name / category
    for root, dirs, files in os.walk(root_dir):
        # Prendiamo il nome della cartella corrente
        category = os.path.basename(root).lower()
        
        if category in valid_categories:
            for file in files:
                if file.endswith('.mp4'):
                    # Creiamo l'entry nel formato compatibile con i tuoi scan
                    entry = {
                        "video_name": file,
                        "category": category
                    }
                    ground_truth.append(entry)
                    
    return ground_truth

def check_duplicates(root_dir):
    # Dizionario che mappa nome_file -> lista delle cartelle (categorie) in cui appare
    file_map = defaultdict(list)
    valid_categories = {'person', 'animal', 'nothing', 'vehicle'}

    for root, dirs, files in os.walk(root_dir):
        category = os.path.basename(root).lower()
        if category in valid_categories:
            for file in files:
                if file.endswith('.mp4'):
                    file_map[file].append(category)

    # Filtriamo solo quelli che compaiono più di una volta
    duplicates = {name: cats for name, cats in file_map.items() if len(cats) > 1}

    if not duplicates:
        print("✅ Nessun duplicato trovato! Il conteggio extra potrebbe dipendere da file non-video o sottocartelle inaspettate.")
    else:
        print(f"⚠️ Trovati {len(duplicates)} video duplicati:\n")
        for name, cats in duplicates.items():
            print(f"FILE: {name} -> Presente in: {', '.join(cats)}")
            
    print(f"\nConteggio totale file unici: {len(file_map)}")
    print(f"Conteggio totale entry (con duplicati): {sum(len(c) for c in file_map.values())}")

if __name__ == "__main__":
    # Cambia questo percorso con la cartella dove hai smistato i video
    # Esempio: "/home/lordkorost/progetti/smart-surveillance-sorter/2026-02-13/video_smistati"
    path_video = "./2026-02-13/sorted" 
    
    if len(sys.argv) > 1:
        path_video = sys.argv[1]

    if not os.path.exists(path_video):
        print(f"Errore: Il percorso {path_video} non esiste.")
        sys.exit(1)

    risultati = genera_ground_truth(path_video)
    
    output_file = "ground_truth.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(risultati, f, indent=4, ensure_ascii=False)
    #check_duplicates(path_video)
    print(f"Fatto! Generato {output_file} con {len(risultati)} video mappati.")