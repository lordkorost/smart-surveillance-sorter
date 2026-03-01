🛡️ Smart Surveillance Sorter

Basta falsi positivi. Organizza i video del tuo NVR usando la potenza di YOLO, CLIP, BLIP e modelli Vision (Ollama).

Pensato per chi riceve centinaia di registrazioni inutili dovute a vento, insetti o foglie, questo tool esamina ogni video e lo cataloga automaticamente in: PERSON, ANIMAL, VEHICLE o OTHERS.

✨ Key Features

    Pipeline Ibrida: YOLO per la velocità → CLIP+BLIP per la precisione → Vision (Ollama) per i casi dubbi (opzionale).

    Early Exit Intelligent: Se rileva una persona in modo certo, interrompe l'analisi risparmiando tempo e GPU.

    Real-Time & Batch: Funziona sia su cartelle storiche che in monitoraggio costante mentre l'NVR salva i file.

    Privacy Totale: Tutto gira in locale. Nessun dato viene inviato all'esterno.

    Resiliente: Resume automatico in ogni fase. Se la corrente salta, riparte esattamente da dove si era fermato.

🔧 Requisiti 

    OS: Linux (Testato su Ubuntu 24.04 con gpu amd rx 9060 xt)

    Python: 3.12

    Hardware: 16GB RAM | 8GB VRAM (minimo)

    AI: Ollama con modello qwen3-vl:8b (per la modalità Vision opzionale)

    Accelerazione: Supporto completo a CUDA e ROCm (AMD) o CPU 

🚀 Quick Start

    Avvia la webui o modifica il settings.json e cameras.json nella cartella /config

    Configura la tua città per il calcolo automatico alba/tramonto (per le soglie giorno/notte).

    Mappa il tuo NVR: Adatta il filename_template in settings.json al formato dei file del tuo NVR.

    Definisci le Camere: Configura cameras.json (vedi cameras_example.json).

    Attenzione: i parametri di default sono stati configurati per telecamere Reolink 4k da esterno, esegui delle scansioni di test e adatta le confidenze di yolo alle tue telecamere per una migliore precisione (Tab test della webui o main.py --test)


📖 Documentazione Approfondita

Per non rendere questo README infinito, abbiamo diviso i dettagli in sezioni specifiche:

    Logica di Scansione & Early Exit: Perché il sistema è veloce e come decide quando smettere di analizzare un video.

    Tuning & False Positives: Come usare i Fake Weights (es. legno, foglie) e la descrizione delle camere per istruire l'AI.

    Modalità Real-Time & Resume: Come configurare il monitoraggio continuo della cartella FTP/Samba.

    Test & Benchmarking: Come usare le modalità --test, --no-sort e --compare per affinare i parametri.

💡 Tips veloci

    Nani da giardino? Descrivili nella desc della telecamera, l'AI smetterà di scambiarli per persone.

    Troppi veicoli? Usa ignore_labels: ["car", "truck"] per quella specifica telecamera.

    Gatti notturni mancati? Abbassa THRESHOLD_ANIMAL nelle thresholds_night.

Install:

git clone ...
cd smart-surveillance-sorter
./install.sh          # Linux (auto-rileva GPU)
./install.sh --use-rocm   # forza ROCm
./install.sh --use-cuda   # forza CUDA
./run.sh              # avvia WebUI


**Sezione risultati/benchmark** — è la cosa che convince chi trova il progetto:

Dataset 26 Feb 2026 — 426 video, telecamere Reolink 4K outdoor

| Mode          | Precision | Recall | Speed      |
|---------------|-----------|--------|------------|
| YOLO+BLIP     | 97.3%     | 100%   | 5.4s/vid   |
| YOLO+Vision   | 98.2%     | 100%   | 5.4+2.8s   |


**Struttura `/docs`** che ti consiglio:
docs/
  scanning-logic.md      # pipeline, early exit, stride
  tuning-guide.md        # fake weights, thresholds, tips
  realtime-resume.md     # real-time sorter, resume
  benchmarks.md          # tutti i tuoi test con numeri
  cameras-config.md      # come configurare cameras.json


  📂 Sezione: Accuracy Testing & Comparison Tool
Il progetto include una suite di strumenti per misurare oggettivamente le performance del sistema:
    1. Ground Truth Generator: Uno script dedicato per creare il dataset di riferimento partendo da file smistati manualmente.
    2. Comparison Engine (--compare): Confronta il file dei risultati generato dal sorter con la Ground Truth e genera automaticamente le metriche:
        ◦ TP (True Positives): Rilevamenti corretti.
        ◦ FP (False Positives): Errori di "eccesso" (ha visto qualcosa che non c'era).
        ◦ FN (False Negatives): Rilevamenti mancati (il cane perso).
        ◦ Precision & Recall: Per avere una percentuale chiara dell'affidabilità.
        
Da webui o python main.py --dir /percorso/test --compare

