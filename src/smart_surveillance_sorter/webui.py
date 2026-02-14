import logging
import os
import signal
import time
import gradio as gr
import json
from pathlib import Path

# Import dal tuo progetto
from smart_surveillance_sorter.constants import CAMERAS_JSON, SETTINGS_JSON, MODELS_DIR, PROMPTS_JSON
from smart_surveillance_sorter.log_config import configure_logger
from smart_surveillance_sorter.scanners.scanner import Scanner
from smart_surveillance_sorter.scanners.vision_helpers import build_dynamic_prompt
from smart_surveillance_sorter.utils import load_json, save_json, validate_ollama_setup

# --- FUNZIONI DI SERVIZIO ---


def ui_validate_ollama(ip, port, model_name):
    # Costruiamo il dizionario che la tua funzione si aspetta
    mock_settings = {
        "model_name": model_name,
        "ollama_conf": {
            "ip": ip,
            "port": port
        }
    }
    
    # Chiamiamo la tua funzione originale
    is_valid = validate_ollama_setup(mock_settings)
    
    if is_valid:
        return "✅ Validazione riuscita! Ollama risponde correttamente."
    else:
        return "❌ Validazione fallita. Controlla che Ollama sia attivo e il modello caricato."

def preview_prompt_logic(sys_inst, rules, d_p, d_a, d_v, m_c, m_f, mode, has_crop, is_fallback):
    # 1. Ricostruiamo al volo un dizionario prompts_config basato sulla UI
    temp_prompts_config = {
        "shared_components": {
            "system_instruction": sys_inst,
            "mandatory_rules": rules
        },
        "class_descriptions": {
            "PERSON": d_p,
            "ANIMAL": d_a,
            "VEHICLE": d_v
        },
        "modules": {
            "analyst_mission_crop": m_c,
            "fallback_header": m_f
        },
        "templates": load_json(PROMPTS_JSON).get("templates", {}) # Prendiamo i template dal file (che non cambiano)
    }

    # 2. Simuliamo una cam_cfg di test
    test_cam_cfg = {
        "desc": "ZONA TEST: Ingresso principale con giardino e parcheggio.",
        "filters": {"ignore_labels": []}
    }

    try:
        # 3. Chiamiamo la tua funzione originale!
        final_prompt = build_dynamic_prompt(
            temp_prompts_config, 
            test_cam_cfg, 
            mode=mode, 
            has_crop=has_crop, 
            is_fallback=is_fallback
        )
        return final_prompt
    except Exception as e:
        return f"❌ Errore nella costruzione del prompt: {str(e)}"
    

def delete_camera(cam_id):
    if not cam_id:
        return gr.update(), "⚠️ Seleziona una telecamera da eliminare."
    
    cameras = load_json(CAMERAS_JSON)
    
    if cam_id in cameras:
        cam_name = cameras[cam_id].get("name", "Sconosciuta")
        del cameras[cam_id]
        save_json(cameras, CAMERAS_JSON)
        
        # Prepariamo la nuova lista per il dropdown
        new_ids = list(cameras.keys())
        new_val = new_ids[0] if new_ids else None
        
        # Restituiamo l'aggiornamento del dropdown e un messaggio di stato
        return gr.update(choices=new_ids, value=new_val), f"🗑️ Telecamera {cam_id} ({cam_name}) eliminata."
    
    return gr.update(), "❌ Errore: Telecamera non trovata."

def update_fallback_availability(engine):
    # Se l'engine è vision, fallback è abilitato. Se è blip, lo disabilitiamo.
    if engine == "vision":
        return gr.update(interactive=True, value=True)
    else:
        # Lo disattiviamo e togliamo la spunta per coerenza
        return gr.update(interactive=False, value=False)
def update_engines_availability(engine):
    """
    Gestisce l'interattività dei checkbox in base all'engine selezionato.
    Il check_clean e il fallback avanzato richiedono 'vision' (Ollama).
    """
    if engine == "vision":
        # Abilitiamo tutto
        return (
            gr.update(interactive=True, value=True),  # fallback
            gr.update(interactive=True)               # check_clean
        )
    else:
        # Disabilitiamo e togliamo la spunta per coerenza
        return (
            gr.update(interactive=False, value=False), # fallback
            gr.update(interactive=False, value=False)  # check_clean
        )

def add_new_camera():
    cameras = load_json(CAMERAS_JSON)
    # Trova il prossimo ID disponibile (es: se hai 00, 01, crea 02)
    existing_ids = sorted([int(k) for k in cameras.keys()])
    next_id = f"{max(existing_ids) + 1:02d}" if existing_ids else "00"
    
    # Crea un template vuoto
    cameras[next_id] = {"name": "Nuova Cam", "search_patterns": [f"_{next_id}_"], "thresholds": {"person": 0.5, "vehicle": 0.5, "animal": 0.5}}
    save_json(cameras, CAMERAS_JSON)
    
    # Aggiorna il dropdown con la nuova lista
    return gr.update(choices=list(cameras.keys()), value=next_id)

def load_camera_details(cam_id):
    # if not cam_id:
    #     return [None] * 10
    if not cam_id:
        # Se non c'è cam_id (es. lista vuota), svuota tutti i campi
        return [""] * 5 + [False] + [""] + [0.5] * 3
    cameras = load_json(CAMERAS_JSON)
    cam = cameras.get(cam_id, {})
    
    # Prepariamo i valori (gestendo i default se mancano chiavi)
    return [
        cam.get("name", ""),
        cam.get("location", ""),
        ", ".join(cam.get("search_patterns", [])),
        cam.get("priority", "person"),
        cam.get("desc", ""),
        cam.get("dynamic_stride", False),
        ", ".join(cam.get("filters", {}).get("ignore_labels", [])),
        cam.get("thresholds", {}).get("person", 0.5),
        cam.get("thresholds", {}).get("vehicle", 0.5),
        cam.get("thresholds", {}).get("animal", 0.5)
    ]

def save_single_camera(cam_id, name, loc, patterns, priority, desc, dynamic, ignore, th_p, th_v, th_a):
    cameras = load_json(CAMERAS_JSON)
    
    cameras[cam_id] = {
        "name": name,
        "location": loc,
        "search_patterns": [p.strip() for p in patterns.split(",")],
        "priority": priority,
        "desc": desc,
        "dynamic_stride": dynamic,
        "filters": {"ignore_labels": [i.strip() for i in ignore.split(",") if i.strip()]},
        "thresholds": {"person": th_p, "vehicle": th_v, "animal": th_a}
    }
    
    save_json(cameras, CAMERAS_JSON)
    return f"✅ Telecamera {cam_id} ({name}) salvata!"

def save_prompts_ui(sys_inst, rules, d_p, d_a, d_v, d_clean, m_c, m_f, m_clean):
    try:
        # Carichiamo il file originale per preservare i templates
        data = load_json(PROMPTS_JSON)
        
        # Aggiorniamo solo i componenti testuali
        data["shared_components"]["system_instruction"] = sys_inst
        data["shared_components"]["mandatory_rules"] = rules
        data["class_descriptions"]["PERSON"] = d_p
        data["class_descriptions"]["ANIMAL"] = d_a
        data["class_descriptions"]["VEHICLE"] = d_v
        data["modules"]["analyst_mission_crop"] = m_c
        data["modules"]["fallback_header"] = m_f
        
        # Salvataggio
        save_json(data, PROMPTS_JSON)
        return "✅ Prompt AI aggiornati! I cambiamenti influenzeranno le prossime analisi."
    except Exception as e:
        return f"❌ Errore nel salvataggio dei prompt: {str(e)}"


    
def load_configs():
    """Carica i file JSON direttamente senza utility esterne."""
    print(f"DEBUG: Lettura diretta file...")
    
    # --- Caricamento Settings ---
    try:
        if SETTINGS_JSON.exists():
            with open(SETTINGS_JSON, "r", encoding="utf-8") as f:
                settings_raw = json.load(f)
                settings_str = json.dumps(settings_raw, indent=4, ensure_ascii=False)
                print(f"✅ DEBUG: settings.json letto ({len(settings_str)} caratteri)")
        else:
            settings_str = "{ 'info': 'File settings.json non trovato' }"
    except Exception as e:
        print(f"❌ DEBUG ERRORE Settings: {e}")
        settings_str = f"{{ 'error': '{str(e)}' }}"

    # --- Caricamento Cameras ---
    try:
        if CAMERAS_JSON.exists():
            with open(CAMERAS_JSON, "r", encoding="utf-8") as f:
                cameras_raw = json.load(f)
                cameras_str = json.dumps(cameras_raw, indent=4, ensure_ascii=False)
                print(f"✅ DEBUG: cameras.json letto ({len(cameras_str)} caratteri)")
        else:
            cameras_str = "{ 'info': 'File cameras.json non trovato' }"
    except Exception as e:
        print(f"❌ DEBUG ERRORE Cameras: {e}")
        cameras_str = f"{{ 'error': '{str(e)}' }}"

    try:
        if PROMPTS_JSON.exists():
            with open(PROMPTS_JSON, "r", encoding="utf-8") as f:
                prompt_raw = json.load(f)
                prompt_str = json.dumps(prompt_raw, indent=4, ensure_ascii=False)
                print(f"✅ DEBUG: prompts.json letto ({len(cameras_str)} caratteri)")
        else:
            cameras_str = "{ 'info': 'File prompts.json non trovato' }"
    except Exception as e:
        print(f"❌ DEBUG ERRORE promps: {e}")
        cameras_str = f"{{ 'error': '{str(e)}' }}"

    return settings_str, cameras_str,prompt_str

def save_config_ui(content, file_path):
    """Salva il contenuto dell'editor nel file JSON."""
    try:
        data = json.loads(content)
        success = save_json(data, file_path)
        if success:
            return f"✅ Salvato con successo in: {file_path.name} ({time.strftime('%H:%M:%S')})"
        return f"❌ Errore durante il salvataggio di {file_path.name}"
    except Exception as e:
        return f"❌ Errore nel formato JSON: {str(e)}"

def get_available_models():
    """Elenca i modelli YOLO disponibili."""
    if MODELS_DIR.exists():
        models = [f.name for f in MODELS_DIR.glob("*.pt")]
        return models if models else ["yolov8l.pt"]
    return ["yolov8l.pt"]


def shutdown_server():
    print("🛑 Spegnimento della WebUI in corso...")
    # Aspettiamo un secondo per permettere alla UI di mostrare il messaggio
    os.kill(os.getpid(), signal.SIGINT) 
    return "Server arrestato. Puoi chiudere questa scheda."

def save_comprehensive_settings(*args):
    # args contiene tutti i valori della UI nell'ordine in cui sono messi negli 'inputs'
    try:
        data = load_json(SETTINGS_JSON)
        
        # Mappatura manuale semplificata (seguendo l'ordine degli inputs del bottone)
        (priority, save_others, fn_temp, struct, y_mod, y_dev, y_stri, y_occ, y_gap, 
         th_p, th_v, th_a, v_mod, v_temp, o_ip, o_port, v_tk, v_tp, 
         w_h, w_m, w_l, sc_p, sc_a, sc_v) = args

        # Update dict
        data["classification_settings"]["priority_hierarchy"] = [x.strip() for x in priority.split(",")]
        data["classification_settings"]["save_others"] = save_others
        data["storage_settings"]["filename_template"] = fn_temp
        data["storage_settings"]["structure_type"] = struct
        data["yolo_settings"]["model_path"] = y_mod
        data["yolo_settings"]["device"] = y_dev
        data["yolo_settings"]["vid_stride"] = int(y_stri)
        data["yolo_settings"]["num_occurrence"] = int(y_occ)
        data["yolo_settings"]["time_gap_sec"] = int(y_gap)
        data["yolo_settings"]["thresholds"] = {"person": th_p, "vehicle": th_v, "animal": th_a}
        
        data["vision_settings"]["model_name"] = v_mod
        data["vision_settings"]["temperature"] = v_temp
        data["vision_settings"]["ollama_conf"] = {"ip": o_ip, "port": o_port}
        data["vision_settings"]["top_k"] = int(v_tk)
        data["vision_settings"]["top_p"] = v_tp
        
        data["scoring_system"]["weights"] = {"score_high": w_h, "score_mid": w_m, "score_low": w_l}
        data["scoring_system"]["thresholds"] = {"person": sc_p, "animal": sc_a, "vehicle": sc_v}

        save_json(data, SETTINGS_JSON)
        return "✅ Tutte le impostazioni salvate! Lo Scanner userà questi valori al prossimo avvio."
    except Exception as e:
        return f"❌ Errore nel salvataggio: {str(e)}"

# Usiamo debug=True di default per la UI o lo aggiorniamo nel run_process
configure_logger(debug=True) 
log = logging.getLogger(__name__)

def run_process(input_path, output_path, mode, model_name, use_refine, use_fallback, test_mode, engine, device,is_clean_check):
    """Avvia lo Scanner e riporta i log alla UI."""
    if not input_path:
        yield "⚠️ Errore: Devi specificare almeno la cartella di input."
        return
    
    # --- AGGIUNGI QUESTA RIGA ---
    input_p = Path(input_path) 
    
    # Ora usa 'input_p' invece di 'input_path' per i controlli
    if not input_p.exists() or not input_p.is_dir():
        yield f"❌ Errore: Il percorso '{input_path}' non esiste o non è una cartella."
        return
    configure_logger(debug=test_mode)
    log.info(f"🚀 Avvio scansione da WebUI (Mode: {mode}, Engine: {engine}, Test: {test_mode})")
    final_output = output_path if (output_path and output_path.strip()) else input_path
    
    yield f"🚀 Inizializzazione Scanner...\n📂 In: {input_path}\n📂 Out: {final_output}"
    # 2. AGGIORNAMENTO DINAMICO SETTINGS (yolo_settings -> model_path)
    try:
        with open(SETTINGS_JSON, "r", encoding="utf-8") as f:
            settings_dict = json.load(f)
        
        # Puliamo l'estensione .pt se presente nel dropdown per scriverlo come piace a te
        clean_model_name = model_name.replace(".pt", "")
        
        # Aggiorniamo i campi in base alla tua struttura
        if "yolo_settings" not in settings_dict:
            settings_dict["yolo_settings"] = {}
        
        settings_dict["yolo_settings"]["model_path"] = clean_model_name
        settings_dict["yolo_settings"]["device"] = device if device else "cuda"
        
        # Salviamo il file aggiornato
        with open(SETTINGS_JSON, "w", encoding="utf-8") as f:
            json.dump(settings_dict, f, indent=4, ensure_ascii=False)
            
        print(f"✅ DEBUG UI: settings.json aggiornato (Model: {clean_model_name}, Device: {device})")
    except Exception as e:
        print(f"⚠️ DEBUG UI: Errore durante l'aggiornamento dei settings: {e}")
    try:

        start_time = time.time()
        scanner = Scanner(
            mode=mode,
            device=device if device else None, # Passiamo il device (cpu, cuda, etc)
            is_refine=use_refine,
            is_fallback=use_fallback,
            is_test=test_mode,
            engine=engine,
            is_check_clean=is_clean_check # <--- Fondamentale per il 117!
        )

        # Esecuzione (i log dettagliati usciranno nel terminale grazie al logger root)
        scanner.scan_folder(input_path, final_output)
        
        elapsed = time.time() - start_time
        
        # Invece di stampare tutto, recuperiamo il riepilogo finale
        # Se hai accesso alla funzione che genera le statistiche, puoi catturarla
        summary = f"✅ ANALISI COMPLETATA in {elapsed:.2f}s!\n\n"
        summary += "📊 Controlla il terminale per il riepilogo dettagliato (117/117)."
        
        yield summary
        scanner._print_final_summary(time.time() - start_time)

    except Exception as e:
        yield f"💥 Errore durante lo scan: {str(e)}"
        # yield "🔍 Scansione avviata... (Controlla il terminale per la barra di progresso)"
        
        # start_time = time.time()
        # # Eseguiamo lo scan
        # scanner.scan_folder(input_path, final_output)
        # elapsed = time.time() - start_time
        
        # yield f"✅ COMPLETATO in {elapsed:.2f}s!\n📂 Risultati salvati in: {final_output}"

    except Exception as e:
        import traceback
        yield f"❌ ERRORE DURANTE LO SCAN:\n{str(e)}\n\n{traceback.format_exc()}"

# --- INTERFACCIA ---
init_set, init_cam, init_prompt = load_configs()
with gr.Blocks(title="Smart Surveillance Sorter UI") as demo:
    gr.Markdown("# 🛡️ Smart Surveillance Sorter")
    
    with gr.Tabs():
        # --- TAB 1: CONTROLLO ---
        with gr.TabItem("🚀 Run Scan"):
            with gr.Row():
                with gr.Column():
                    input_path = gr.Textbox(label="Input Directory (es: /home/user/fails)", placeholder="Percorso cartella da scansionare...")
                    output_path = gr.Textbox(label="Output Directory (opzionale)", placeholder="Se vuoto, usa la cartella di input...")
                    
                    mode_opt = gr.Radio(["full", "person", "person_animal"], label="Modalità", value="full")
                    is_test = gr.Checkbox(label="🧪 Modalità TEST (test_metrics.json)", value=True)

                with gr.Column():
                  
                    # Dropdown per il modello come prima
                    yolo_model = gr.Dropdown(
                        choices=get_available_models(), 
                        label="Modello YOLO", 
                        value="yolov8l.pt"
                    )
                    
                    # Sostituiamo la Textbox con il Radio button
                    device_opt = gr.Radio(
                        choices=["cuda", "cpu", "mps"], 
                        label="Device (Hardware Acceleration)", 
                        value="cuda", # Default consigliato se hai una GPU
                        interactive=True
    )
                    
                    with gr.Group():
                        gr.Markdown("### 🧠 AI Engines & Maintenance")
                        
                        # --- NUOVO CHECK-CLEAN ---
                        check_clean = gr.Checkbox(
                            label="🕸️ Check Lens Health (Spider Webs/Dirt)", 
                            value=False,
                            info="Confronta i frame correnti con le immagini di riferimento in 'checks/'"
                        )
                        
                        with gr.Row():
                            refine = gr.Checkbox(label="✨ Enable Refine", value=True)
                            fallback = gr.Checkbox(label="🔍 Fallback Recovery", value=True)
                        
                        engine_opt = gr.Radio(
                            choices=["vision", "blip"], 
                            label="Refinement Engine", 
                            value="vision",
                            interactive=True
                        )
                        
                        # Logica di attivazione fallback (invariata)
                        engine_opt.change(
                            fn=update_engines_availability,
                            inputs=engine_opt,
                            outputs=[fallback, check_clean] # Aggiorna entrambi!
                        )
                        gr.Markdown("_Nota: 'vision' usa Ollama/Qwen, 'blip' usa il modello locale._")
            
            run_btn = gr.Button("🔥 AVVIA SCANSIONE", variant="primary", size="lg")
            output_log = gr.Textbox(label="Status Log", interactive=False, lines=8)

            run_btn.click(
                fn=run_process,
                inputs=[
                    input_path, 
                    output_path, 
                    mode_opt, 
                    yolo_model, 
                    refine, 
                    fallback, 
                    is_test, 
                    engine_opt,  # <--- NUOVO
                    device_opt,   # <--- NUOVO
                    check_clean
                ],
                outputs=output_log
)

        # --- TAB 2: CONFIGURAZIONE ---
        #with gr.TabItem("⚙️ Configurazione"):
        #with gr.TabItem("⚙️ Configurazione"):
        with gr.TabItem("⚙️ Configurazione"):
            current_settings = load_json(SETTINGS_JSON)

            with gr.Tabs():
                with gr.Tab("🛠️ Settings Generali"):
                    
                    # --- 1. CLASSIFICAZIONE & STORAGE ---
                    with gr.Accordion("📂 Classificazione e Archiviazione", open=True):
                        with gr.Row():
                            priority = gr.Textbox(label="Gerarchia Priorità", 
                                                value=", ".join(current_settings["classification_settings"]["priority_hierarchy"]))
                            save_others = gr.Checkbox(label="Salva 'Others'", 
                                                    value=current_settings["classification_settings"]["save_others"])
                        with gr.Row():
                            fn_template = gr.Textbox(label="Template Nome File", value=current_settings["storage_settings"]["filename_template"])
                            struct_type = gr.Dropdown(choices=["camera_first", "date_first"], label="Struttura Cartelle", 
                                                    value=current_settings["storage_settings"]["structure_type"])

                    # --- 2. YOLO SETTINGS ---
                    with gr.Accordion("🤖 YOLO (Rilevamento Locale)", open=False):
                        with gr.Row():
                            y_mod = gr.Textbox(label="Model Path", value=current_settings["yolo_settings"]["model_path"])
                            y_dev = gr.Radio(choices=["cuda", "cpu", "mps"], label="Device", value=current_settings["yolo_settings"]["device"])
                        with gr.Row():
                            y_stri = gr.Number(label="Video Stride", value=current_settings["yolo_settings"]["vid_stride"])
                            y_occ = gr.Slider(1, 10, step=1, label="Num. Occorrenze", value=current_settings["yolo_settings"]["num_occurrence"])
                            y_gap = gr.Number(label="Time Gap (sec)", value=current_settings["yolo_settings"]["time_gap_sec"])
                        
                        gr.Markdown("#### 🎯 Soglie YOLO")
                        with gr.Row():
                            th_p = gr.Slider(0, 1, value=current_settings["yolo_settings"]["thresholds"]["person"], label="Soglia Persona")
                            th_v = gr.Slider(0, 1, value=current_settings["yolo_settings"]["thresholds"]["vehicle"], label="Soglia Veicolo")
                            th_a = gr.Slider(0, 1, value=current_settings["yolo_settings"]["thresholds"]["animal"], label="Soglia Animale")

                    # --- 3. VISION & OLLAMA ---
                    with gr.Accordion("👁️ Vision AI (Ollama/Qwen)", open=False):
                        with gr.Row():
                            v_mod = gr.Textbox(label="Nome Modello Ollama", value=current_settings["vision_settings"]["model_name"])
                            v_temp = gr.Slider(0, 2, value=current_settings["vision_settings"]["temperature"], label="Temperatura")
                        with gr.Row():
                            ollama_ip = gr.Textbox(label="Ollama IP", value=current_settings.get("vision_settings", {}).get("ollama_conf", {}).get("ip", "127.0.0.1"))
                            ollama_port = gr.Textbox(label="Ollama Port", value=current_settings.get("vision_settings", {}).get("ollama_conf", {}).get("port", "11434"))
                            # --- IL TASTO CHE USA LA TUA FUNZIONE ---
                            with gr.Row():
                                validate_btn = gr.Button("🔍 Test Connessione Ollama", variant="secondary")
                                validate_status = gr.Markdown("Stato: _Non verificato_")
                        with gr.Row():
                            v_topk = gr.Number(label="Top K", value=current_settings["vision_settings"]["top_k"])
                            v_topp = gr.Slider(0, 1, value=current_settings["vision_settings"]["top_p"], label="Top P")

                    # --- 4. SCORING SYSTEM ---
                    with gr.Accordion("⚖️ Scoring System", open=False):
                        with gr.Row():
                            w_high = gr.Number(label="Peso High", value=current_settings["scoring_system"]["weights"]["score_high"])
                            w_mid = gr.Number(label="Peso Mid", value=current_settings["scoring_system"]["weights"]["score_mid"])
                            w_low = gr.Number(label="Peso Low", value=current_settings["scoring_system"]["weights"]["score_low"])
                        with gr.Row():
                            sc_p = gr.Number(label="Min Persona", value=current_settings["scoring_system"]["thresholds"]["person"])
                            sc_a = gr.Number(label="Min Animale", value=current_settings["scoring_system"]["thresholds"]["animal"])
                            sc_v = gr.Number(label="Min Veicolo", value=current_settings["scoring_system"]["thresholds"]["vehicle"])

                    save_all_btn = gr.Button("💾 SALVA TUTTE LE CONFIGURAZIONI", variant="primary", size="lg")
                    status_save = gr.Markdown("")

                    validate_btn.click(
                        fn=ui_validate_ollama,
                        inputs=[ollama_ip, ollama_port, v_mod],
                        outputs=validate_status
                    )
                    # ORA IL CLICK È ALLINEATO:
                    save_all_btn.click(
                        fn=save_comprehensive_settings,
                        inputs=[
                            priority, save_others, fn_template, struct_type, # <--- Corretti struct_type e fn_template
                            y_mod, y_dev, y_stri, y_occ, y_gap,              # <--- Corretti y_mod, y_stri
                            th_p, th_v, th_a, 
                            v_mod, v_temp, ollama_ip, ollama_port, v_topk, v_topp, 
                            w_high, w_mid, w_low, sc_p, sc_a, sc_v
                        ],
                        outputs=status_save
                    )

                # --- SOTTO-TAB: CAMERAS & PROMPTS (Possiamo lasciarli JSON o fare liste dinamiche) ---
                #with gr.Tab("📹 Cameras"):
                with gr.Tab("📹 Gestione Telecamere"):
                    current_cameras = load_json(CAMERAS_JSON)
                    camera_ids = list(current_cameras.keys())

                    with gr.Row():
                        # Selettore della telecamera
                        cam_selector = gr.Dropdown(
                            choices=camera_ids, 
                            label="Seleziona ID Telecamera", 
                            value=camera_ids[0] if camera_ids else None,
                            interactive=True
                        )
                        add_cam_btn = gr.Button("➕ Aggiungi Nuova Cam", variant="secondary")

                    with gr.Group():
                        gr.Markdown("### 📝 Modifica Parametri Telecamera")
                        with gr.Row():
                            c_name = gr.Textbox(label="Nome (es: Orto)")
                            c_loc = gr.Textbox(label="Location (es: outdoor)")
                        
                        with gr.Row():
                            # I pattern li gestiamo come stringa separata da virgola
                            c_patterns = gr.Textbox(label="Search Patterns (separati da virgola)", 
                                                placeholder="_00_, ch00, Cam00")
                            c_priority = gr.Dropdown(choices=["person", "animal", "vehicle"], label="Priorità")
                        
                        c_desc = gr.Textbox(label="Descrizione", lines=2)
                        
                        with gr.Row():
                            c_dynamic = gr.Checkbox(label="Dynamic Stride")
                            c_ignore = gr.Textbox(label="Labels da Ignorare (es: car, truck)")

                        with gr.Row():
                            cth_p = gr.Slider(0, 1, label="Soglia Persona (Cam)")
                            cth_v = gr.Slider(0, 1, label="Soglia Veicolo (Cam)")
                            cth_a = gr.Slider(0, 1, label="Soglia Animale (Cam)")

                    save_cam_btn = gr.Button("💾 Salva Modifiche Telecamera", variant="primary")
                    delete_cam_btn = gr.Button("🗑️ Elimina Telecamera", variant="stop")
                    status_cam = gr.Markdown("")

                    delete_cam_btn.click(
                        fn=delete_camera,
                        inputs=cam_selector,
                        outputs=[cam_selector, status_cam]
                    )
                    save_cam_btn.click(
                        fn=save_single_camera,
                        inputs=[cam_selector, c_name, c_loc, c_patterns, c_priority, c_desc, c_dynamic, c_ignore, cth_p, cth_v, cth_a],
                        outputs=status_cam
                    )
                    add_cam_btn.click(fn=add_new_camera, outputs=cam_selector)
                    cam_selector.change(
                        fn=load_camera_details, # La funzione che legge il JSON (assicurati che sia definita)
                        inputs=cam_selector,
                        outputs=[c_name, c_loc, c_patterns, c_priority, c_desc, c_dynamic, c_ignore, cth_p, cth_v, cth_a]
                    )
                    demo.load(
                        fn=load_camera_details,
                        inputs=cam_selector,
                        outputs=[c_name, c_loc, c_patterns, c_priority, c_desc, c_dynamic, c_ignore, cth_p, cth_v, cth_a]
                    )

                #with gr.Tab("📝 Prompts"):
                with gr.Tab("📝 Editor Prompt AI"):
                    current_prompts = load_json(PROMPTS_JSON)
                    
                    with gr.Accordion("📢 Istruzioni di Sistema (Globali)", open=True):
                    
                        gr.Markdown("_Queste regole valgono per ogni analisi effettuata dall'AI._")
                        p_sys = gr.Textbox(
                            label="System Instruction", 
                            value=current_prompts["shared_components"]["system_instruction"]
                        )
                        p_rules = gr.Textbox(
                            label="Mandatory Rules (Regole Ferree)", 
                            value=current_prompts["shared_components"]["mandatory_rules"], 
                            lines=3
                        )
                    #with gr.Accordion("🎯 Descrizione Classi (Cosa deve cercare)", open=True):
                    with gr.Accordion("🎯 Descrizione Classi (Cosa deve cercare)", open=True):
                        gr.Markdown("_Definisci i criteri con cui l'AI riconosce i soggetti o lo stato delle lenti._")
                        with gr.Row():
                            desc_p = gr.Textbox(label="PERSON Description", value=current_prompts["class_descriptions"]["PERSON"], lines=2)
                        with gr.Row():
                            desc_a = gr.Textbox(label="ANIMAL Description", value=current_prompts["class_descriptions"]["ANIMAL"], lines=2)
                        with gr.Row():
                            desc_v = gr.Textbox(label="VEHICLE Description", value=current_prompts["class_descriptions"]["VEHICLE"], lines=2)
                        # --- NUOVO CAMPO CLEAN CHECK ---
                        with gr.Row():
                            desc_clean = gr.Textbox(label="LENS CLEANING Description", value=current_prompts["class_descriptions"]["CLEAN_CHECK"], lines=3)

                    with gr.Accordion("🧩 Moduli e Intestazioni (Avanzate)", open=False):
                        gr.Markdown("_Testi aggiunti in caso di Crop, Fallback o Maintenance._")
                        m_crop = gr.Textbox(label="Mission Crop Header", value=current_prompts["modules"]["analyst_mission_crop"], lines=3)
                        m_fall = gr.Textbox(label="Fallback Header", value=current_prompts["modules"]["fallback_header"], lines=2)
                        # --- NUOVO CAMPO HEADER CLEAN ---
                        m_clean = gr.Textbox(label="Clean Check Header", value=current_prompts["modules"]["clean_header"], lines=2)
                        
                        with gr.Accordion("🔍 Anteprima Prompt Dinamico", open=False):
                            gr.Markdown("_Simula come verrà costruito il prompt finale._")
                            with gr.Row():
                                test_mode = gr.Dropdown(choices=["full", "person", "person_animal", "clean_check"], label="Simula Modo", value="full")
                                test_has_crop = gr.Checkbox(label="Simula Crop (Zoom)")
                                test_is_fallback = gr.Checkbox(label="Simula Fallback")
                            
                            preview_btn = gr.Button("🔨 Genera Anteprima", variant="secondary")
                            prompt_preview = gr.Code(label="Prompt Finale inviato alla AI", language="markdown", lines=15)

                    save_prompt_btn = gr.Button("💾 Aggiorna Prompt AI", variant="primary")
                    status_prompt = gr.Markdown("")

                    # --- COLLEGAMENTO EVENTO ---
                    preview_btn.click(
                        fn=preview_prompt_logic,
                        inputs=[p_sys, p_rules, desc_p, desc_a, desc_v, m_crop, m_fall, test_mode, test_has_crop, test_is_fallback],
                        outputs=prompt_preview
                    )
                    # Collegamento
                    save_prompt_btn.click(
                        fn=save_prompts_ui,
                        inputs=[
                            p_sys,      # sys_inst
                            p_rules,    # rules
                            desc_p,     # d_p
                            desc_a,     # d_a
                            desc_v,     # d_v
                            desc_clean, # d_clean (Il nuovo campo della classe)
                            m_crop,     # m_c
                            m_fall,     # m_f
                            m_clean     # m_clean (Il nuovo header)
                        ],
                        outputs=status_prompt
                    )
                with gr.Tab("🖥️ Sistema"):
                    with gr.Row():
                        stop_btn = gr.Button("🛑 Spegni WebUI", variant="stop")
                        #restart_btn = gr.Button("🔄 Riavvia (Soft)", variant="secondary")
                    
                    status_sys = gr.Markdown("Stato Sistema: Attivo")

                # Logica corretta
                stop_btn.click(fn=shutdown_server, inputs=None, outputs=status_sys)
                # Se vuoi che il tasto reset aggiorni anche il messaggio di stato:
                #restart_btn.click(fn=lambda: "### 🔄 Configurazioni ricaricate dai file JSON", inputs=None, outputs=status_sys)

    




# Avvio UI
if __name__ == "__main__":
    # # Avviamo col tema corretto qui per evitare i warning di Gradio 6
    # demo.launch(theme=gr.themes.Soft())
    # if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        prevent_thread_lock=False # Aiuta a gestire meglio la chiusura
    )