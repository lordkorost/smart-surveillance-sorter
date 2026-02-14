import streamlit as st
import os
import json
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

# IMPORTIAMO LE COSTANTI CHE ABBIAMO DEFINITO
from smart_surveillance_sorter.constants import (
    PROJECT_ROOT, 
    CAMERAS_JSON, 
    SETTINGS_JSON, 
    PROMPTS_JSON
)

def select_folder():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    selected_path = filedialog.askdirectory(initialdir=os.getcwd())
    root.destroy()
    return selected_path

st.set_page_config(page_title="AI Surveillance Manager", layout="wide", page_icon="🛡️")

# --- TAB 1: CONFIGURAZIONE ---
tab1, tab2, tab3 = st.tabs(["⚙️ Configurazione", "🚀 Esecuzione", "📊 Risultati"])

with tab1:
    st.header("⚙️ Configurazione Sistema")
    # Qui usiamo CAMERAS_JSON importato dalle costanti
    if CAMERAS_JSON.exists():
        with open(CAMERAS_JSON, "r") as f:
            cameras = json.load(f)
        
        cam_id = st.selectbox("Seleziona Telecamera", list(cameras.keys()))
        # ... resto della logica di salvataggio (usa CAMERAS_JSON per scrivere)
    else:
        st.error(f"File non trovato in: {CAMERAS_JSON}")

# --- TAB 2: ESECUZIONE ---
with tab2:
    st.header("🚀 Avvio Scansione")
    
    # Gestione cartella con session_state
    if 'folder_path' not in st.session_state:
        st.session_state.folder_path = str(PROJECT_ROOT)

    col_path, col_btn = st.columns([4, 1])
    input_dir = col_path.text_input("📁 Cartella da analizzare:", st.session_state.folder_path)
    
    if col_btn.button("Sfoglia..."):
        path = select_folder()
        if path:
            st.session_state.folder_path = path
            st.rerun()

    # Parametri comando
    c1, c2, c3 = st.columns(3)
    scan_mode = c1.selectbox("🎯 Modalità", ["person", "animal", "full"])
    model_choice = c2.selectbox("🧠 Modello YOLO", ["yolov8n", "yolov8m", "yolov8l"])
    use_refine = c3.checkbox("✨ Usa Vision Refine (Ollama)", value=True)

    # Costruiamo il comando usando il main.py nella cartella corretta
    main_script = PROJECT_ROOT / "src" / "surveillance_sorter" / "main.py"
    
    cmd = [
        "python", str(main_script),
        "--dir", input_dir,
        "--mode", scan_mode,
        "--model", model_choice
    ]
    if use_refine: cmd.append("--refine")

    if st.button("🔥 LANCIA ANALISI", use_container_width=True):
        # Logica subprocess.Popen (identica alla tua, ma con cmd corretto)
        st.info(f"Esecuzione: {' '.join(cmd)}")


    # --- TAB 3: VISUALIZZAZIONE RISULTATI ---
# Nella Tab Risultati
with tab3:
    st.header("📊 Analisi Risultati")
    
    # Assicuriamoci che input_dir sia un oggetto Path per gestire bene gli spazi
    base_folder = Path(input_dir)
    target_json = base_folder / "vision_classification.json"
    
    if target_json.exists():
        with open(target_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        st.success(f"Caricati {len(data)} eventi da `{base_folder.name}`")

        # --- Griglia di Visualizzazione ---
        for item in data:
            with st.expander(f"🎥 {item['video_name']} - [{item['category'].upper()}]", expanded=(item['category'] == 'person')):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # COSTRUZIONE PERCORSO IMMAGINE
                    # Se il path nel JSON è relativo (es: frames/immagine.jpg)
                    # lo uniamo alla cartella base selezionata dall'utente
                    relative_frame_path = item.get("frame_priority")
                    
                    if relative_frame_path:
                        # Risolviamo il percorso assoluto
                        img_path = base_folder / relative_frame_path
                        
                        if img_path.exists():
                            st.image(str(img_path), caption=f"Rilevazione: {item['category']}", width='stretch')
                        else:
                            st.error(f"File non trovato: {img_path.name}")
                            st.caption(f"Percorso cercato: {img_path}")
                    else:
                        st.warning("Nessun frame di riferimento nel JSON")

                with col2:
                    st.write(f"**Camera:** {item['camera_id']}")
                    st.write(f"**Video Originale:** `{item['video_path']}`")
                    st.write(f"**Data/Ora:** {item.get('timestamp_analysis', 'N/A')}")
                    
                    if item.get('details'):
                        st.info(f"Analizzati {len(item['details'])} frame totali per questo video.")
    else:
        st.info("In attesa del file `vision_classification.json`...")
        st.caption(f"Percorso monitorato: {target_json}")