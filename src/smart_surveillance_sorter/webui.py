import logging
import os
import queue
import signal
import threading
import time
import json
import traceback
from pathlib import Path

import gradio as gr

from smart_surveillance_sorter.constants import CAMERAS_JSON, CLIP_BLIP_JSON, SETTINGS_JSON, MODELS_DIR, PROMPTS_JSON
from smart_surveillance_sorter.logger import get_logger
from smart_surveillance_sorter.scanners.scanner import Scanner
from smart_surveillance_sorter.scanners.vision_helpers import build_dynamic_prompt
from smart_surveillance_sorter.utils import load_json, save_json, validate_ollama_setup

log = get_logger(debug=True)

SETTINGS_DEFAULT = Path(SETTINGS_JSON).parent / "settings_default.json"
SETTINGS_BACKUP  = Path(SETTINGS_JSON).parent / "settings_backup.json"
ALL_BTNS = [run_btn, test_btn, rt_start_btn]

def disable_btns():
    return [gr.update(interactive=False)] * 3

def enable_btns():
    return [gr.update(interactive=True)] * 3
# ==============================================================================
# UTILITY
# ==============================================================================

def get_available_models():
    if MODELS_DIR.exists():
        models = [f.name for f in MODELS_DIR.glob("*.pt")]
        return models if models else ["yolov8l.pt"]
    return ["yolov8l.pt"]

def shutdown_server():
    print("🛑 WebUI shutdown...")
    os.kill(os.getpid(), signal.SIGINT)
    return "Server stop. You can close this tab."

def load_configs():
    def _load(path):
        try:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    return json.dumps(json.load(f), indent=4, ensure_ascii=False)
        except Exception as e:
            return f"{{ 'error': '{str(e)}' }}"
        return "{ 'info': 'File not found' }"
    return _load(SETTINGS_JSON), _load(CAMERAS_JSON), _load(PROMPTS_JSON)

def save_config_ui(content, file_path):
    try:
        data = json.loads(content)
        if save_json(data, file_path):
            return f"✅ Salvato: {file_path.name} ({time.strftime('%H:%M:%S')})"
        return f"❌ Save error {file_path.name}"
    except Exception as e:
        return f"❌ Error JSON: {str(e)}"

# ==============================================================================
# SETTINGS BACKUP / RESTORE / DEFAULT
# ==============================================================================

def backup_settings():
    """Save a copy of settings.json and clip_blip_settings.json."""
    try:
        import shutil
        shutil.copy2(SETTINGS_JSON, SETTINGS_BACKUP)
        return True
    except Exception as e:
        log.error(f"Backup settings failed: {e}")
        return False


def restore_settings_backup():
    try:
        if not SETTINGS_BACKUP.exists():
            return "⚠️ No backup found.", *([gr.update()] * 4)
        import shutil
        shutil.copy2(SETTINGS_BACKUP, SETTINGS_JSON)
        # Rileggi i valori ripristinati
        s = load_json(SETTINGS_JSON)
        dss = s["yolo_settings"]["dynamic_stride_settings"]
        return (
            "✅ Settings restored from pre-test backup.",
            s["yolo_settings"].get("vid_stride_sec", 0.6),
            dss.get("warmup_sec", 5),
            dss.get("stride_fast_sec", 1.0),
            dss.get("pre_roll_sec", 20),
        )
    except Exception as e:
        return f"❌ Error: {str(e)}", *([gr.update()] * 4)
    
def restore_settings_default():
    try:
        if not SETTINGS_DEFAULT.exists():
            return "⚠️ settings_default.json file not found in config folder."
        import shutil
        shutil.copy2(SETTINGS_DEFAULT, SETTINGS_JSON)
        return "✅ Settings restored to default."
    except Exception as e:
        return f"❌ Default reset error: {str(e)}"

# ==============================================================================
# OLLAMA
# ==============================================================================

def ui_validate_ollama(ip, port, model_name):
    mock_settings = {"model_name": model_name, "ollama_conf": {"ip": ip, "port": port}}
    if validate_ollama_setup(mock_settings):
        return "✅ Ollama responds correctly."
    return "❌ Validation failed. Please check that Ollama is active and the model is downloaded"

# ==============================================================================
# PROMPT
# ==============================================================================

def preview_prompt_logic(sys_inst, rules, d_p, d_a, d_v, m_c, m_f, mode, has_crop, is_fallback):
    temp_prompts_config = {
        "shared_components": {"system_instruction": sys_inst, "mandatory_rules": rules},
        "class_descriptions": {"PERSON": d_p, "ANIMAL": d_a, "VEHICLE": d_v},
        "modules": {"analyst_mission_crop": m_c, "fallback_header": m_f},
        "templates": load_json(PROMPTS_JSON).get("templates", {})
    }
    test_cam_cfg = {"desc": "Test Zone: Main entrance with garden and parking.", "filters": {"ignore_labels": []}}
    try:
        return build_dynamic_prompt(temp_prompts_config, test_cam_cfg, mode=mode, has_crop=has_crop, is_fallback=is_fallback)
    except Exception as e:
        return f"❌ Prompt construction error: {str(e)}"

def save_prompts_ui(sys_inst, rules, d_p, d_a, d_v, d_clean, m_c, m_f, m_clean):
    try:
        data = load_json(PROMPTS_JSON)
        data["shared_components"]["system_instruction"]   = sys_inst
        data["shared_components"]["mandatory_rules"]      = rules
        data["class_descriptions"]["PERSON"]              = d_p
        data["class_descriptions"]["ANIMAL"]              = d_a
        data["class_descriptions"]["VEHICLE"]             = d_v
        data["class_descriptions"]["CLEAN_CHECK"]         = d_clean
        data["modules"]["analyst_mission_crop"]           = m_c
        data["modules"]["fallback_header"]                = m_f
        data["modules"]["clean_header"]                   = m_clean
        save_json(data, PROMPTS_JSON)
        return "✅ Prompt AI aggiornati."
    except Exception as e:
        return f"❌ Errore prompt save: {str(e)}"

# ==============================================================================
# CAMERAS
# ==============================================================================

def load_camera_details(cam_id):
    empty = ["", "", "", "", "", False, "", 0.49, 0.55, 0.30, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    if not cam_id:
        return empty
    cameras = load_json(CAMERAS_JSON)
    cam = cameras.get(cam_id, {})
    br  = cam.get("blip_rules", {})
    tn  = cam.get("thresholds_night", {})
    return [
        cam.get("name", ""),
        cam.get("location", ""),
        ", ".join(cam.get("search_patterns", [])),
        cam.get("priority", "person"),
        cam.get("desc", ""),
        cam.get("dynamic_stride", False),
        ", ".join(cam.get("filters", {}).get("ignore_labels", [])),
        cam.get("thresholds", {}).get("person",  0.49),
        cam.get("thresholds", {}).get("vehicle", 0.55),
        cam.get("thresholds", {}).get("animal",  0.30),
        tn.get("person",  -1.0),
        tn.get("vehicle", -1.0),
        tn.get("animal",  -1.0),
        br.get("FAKE_WEIGHTS", {}).get("GROUND", 1.0),
        br.get("FAKE_WEIGHTS", {}).get("GARDEN", 1.0),
        br.get("FAKE_WEIGHTS", {}).get("SHOE",   1.0),
        br.get("FAKE_WEIGHTS", {}).get("WOOD",   1.0),
        br.get("THRESHOLD", {}).get("PERSON",  -1.0),
        br.get("THRESHOLD", {}).get("VEHICLE", -1.0),
        br.get("THRESHOLD", {}).get("ANIMAL",  -1.0),
        br.get("FAKE_PENALTY_WEIGHT", {}).get("PERSON",  -1.0),
        br.get("FAKE_PENALTY_WEIGHT", {}).get("ANIMAL",  -1.0),
        br.get("FAKE_PENALTY_WEIGHT", {}).get("VEHICLE", -1.0),
    ]

def save_single_camera(cam_id, name, loc, patterns, priority, desc, dynamic, ignore,
                       th_p, th_v, th_a, nth_p, nth_v, nth_a,
                       fw_ground, fw_garden, fw_shoe, fw_wood,
                       thr_person, thr_vehicle, thr_animal,
                       fpw_person, fpw_animal, fpw_vehicle):
    cameras = load_json(CAMERAS_JSON)
    thresholds_night = {}
    if float(nth_p) >= 0: thresholds_night["person"]  = float(nth_p)
    if float(nth_v) >= 0: thresholds_night["vehicle"] = float(nth_v)
    if float(nth_a) >= 0: thresholds_night["animal"]  = float(nth_a)
    threshold_override = {}
    if float(thr_person)  >= 0: threshold_override["PERSON"]  = float(thr_person)
    if float(thr_vehicle) >= 0: threshold_override["VEHICLE"] = float(thr_vehicle)
    if float(thr_animal)  >= 0: threshold_override["ANIMAL"]  = float(thr_animal)
    penalty_override = {}
    if float(fpw_person)  >= 0: penalty_override["PERSON"]  = float(fpw_person)
    if float(fpw_animal)  >= 0: penalty_override["ANIMAL"]  = float(fpw_animal)
    if float(fpw_vehicle) >= 0: penalty_override["VEHICLE"] = float(fpw_vehicle)
    blip_rules = {"FAKE_WEIGHTS": {"GROUND": float(fw_ground), "GARDEN": float(fw_garden),
                                   "SHOE": float(fw_shoe), "WOOD": float(fw_wood)}}
    if threshold_override: blip_rules["THRESHOLD"] = threshold_override
    if penalty_override:   blip_rules["FAKE_PENALTY_WEIGHT"] = penalty_override
    cam_data = {
        "name": name, "location": loc,
        "search_patterns": [p.strip() for p in patterns.split(",") if p.strip()],
        "priority": priority, "desc": desc, "dynamic_stride": dynamic,
        "filters": {"ignore_labels": [i.strip() for i in ignore.split(",") if i.strip()]},
        "thresholds": {"person": float(th_p), "vehicle": float(th_v), "animal": float(th_a)},
        "blip_rules": blip_rules,
    }
    if thresholds_night:
        cam_data["thresholds_night"] = thresholds_night
    cameras[cam_id] = cam_data
    save_json(cameras, CAMERAS_JSON)
    return f"✅ Cam {cam_id} ({name}) saved!"

def add_new_camera():
    cameras = load_json(CAMERAS_JSON)
    existing_ids = sorted([int(k) for k in cameras.keys()])
    next_id = f"{max(existing_ids) + 1:02d}" if existing_ids else "00"
    cameras[next_id] = {
        "name": "Nuova Cam", "search_patterns": [f"_{next_id}_"],
        "thresholds": {"person": 0.49, "vehicle": 0.55, "animal": 0.30},
        "blip_rules": {"FAKE_WEIGHTS": {"GROUND": 1.0, "GARDEN": 1.0, "SHOE": 1.0, "WOOD": 1.0}}
    }
    save_json(cameras, CAMERAS_JSON)
    return gr.update(choices=list(cameras.keys()), value=next_id)

def delete_camera(cam_id):
    if not cam_id:
        return gr.update(), "⚠️ Select a camera."
    cameras = load_json(CAMERAS_JSON)
    if cam_id in cameras:
        cam_name = cameras[cam_id].get("name", "")
        del cameras[cam_id]
        save_json(cameras, CAMERAS_JSON)
        new_ids = list(cameras.keys())
        return gr.update(choices=new_ids, value=new_ids[0] if new_ids else None), f"🗑️ {cam_id} ({cam_name}) deleted."
    return gr.update(), "❌ Camera not found."

# ==============================================================================
# ENGINE AVAILABILITY
# ==============================================================================

def update_engines_availability(engine):
    if engine == "vision":
        return gr.update(interactive=True), gr.update(interactive=True)
    else:
        return gr.update(interactive=True), gr.update(interactive=False, value=False)

# ==============================================================================
# SETTINGS SAVE
# ==============================================================================

def save_comprehensive_settings(*args):
    try:
        (city, priority, save_others, fn_temp, ts_format, struct,
         y_mod, y_dev, y_stride_sec, y_occ, y_gap,
         warmup_sec, stride_fast_sec, pre_roll_sec, cd_sec,
         v_mod, v_temp, o_ip, o_port, v_tk, v_tp, v_num_predict,
         w_h, w_m, w_l, sc_p, sc_a, sc_v, ov_min_conf, ov_min_score,
         cb_w_crop, cb_w_frame, cb_night_boost,
         cb_thr_p, cb_thr_v, cb_thr_a,
         cb_fpw_p, cb_fpw_a, cb_fpw_v,
         cb_blip_p, cb_blip_a, cb_blip_v,
         cb_bbox_ratio, cb_bbox_bonus) = args

        data = load_json(SETTINGS_JSON)
        data["city"] = city
        data["classification_settings"]["priority_hierarchy"] = [x.strip() for x in priority.split(",")]
        data["classification_settings"]["save_others"]        = save_others
        data["storage_settings"]["filename_template"]         = fn_temp
        data["storage_settings"]["timestamp_format"]          = ts_format
        data["storage_settings"]["structure_type"]            = struct
        data["yolo_settings"]["model_path"]                   = y_mod
        data["yolo_settings"]["device"]                       = y_dev
        data["yolo_settings"]["vid_stride_sec"]               = float(y_stride_sec)
        data["yolo_settings"]["num_occurrence"]               = int(y_occ)
        data["yolo_settings"]["time_gap_sec"]                 = int(y_gap)
        data["yolo_settings"]["dynamic_stride_settings"]      = {
            "warmup_sec": int(warmup_sec), "stride_fast_sec": float(stride_fast_sec),
            "pre_roll_sec": int(pre_roll_sec), "cooldown_sec": int(cd_sec),
        }
        data["vision_settings"]["model_name"]  = v_mod
        data["vision_settings"]["temperature"] = float(v_temp)
        data["vision_settings"]["num_predict"] = int(v_num_predict)
        data["vision_settings"]["ollama_conf"] = {"ip": o_ip, "port": o_port}
        data["vision_settings"]["top_k"]       = int(v_tk)
        data["vision_settings"]["top_p"]       = float(v_tp)
        data["scoring_system"]["weights"]      = {"score_high": float(w_h), "score_mid": float(w_m), "score_low": float(w_l)}
        data["scoring_system"]["thresholds"]   = {"person": float(sc_p), "animal": float(sc_a), "vehicle": float(sc_v)}
        data["scoring_system"]["yolo_override"] = {"person_min_conf": float(ov_min_conf), "min_total_score_to_skip_override": float(ov_min_score)}
        save_json(data, SETTINGS_JSON)

        cb_data = load_json(CLIP_BLIP_JSON)
        cb_data["FINAL_WEIGHT_CROP"]       = float(cb_w_crop)
        cb_data["FINAL_WEIGHT_FRAME"]      = float(cb_w_frame)
        cb_data["YOLO_NIGHT_BOOST"]        = float(cb_night_boost)
        cb_data["THRESHOLD"]               = {"PERSON": float(cb_thr_p), "VEHICLE": float(cb_thr_v), "ANIMAL": float(cb_thr_a)}
        cb_data["FAKE_PENALTY_WEIGHT"]     = {"PERSON": float(cb_fpw_p), "ANIMAL": float(cb_fpw_a), "VEHICLE": float(cb_fpw_v)}
        cb_data["BLIP_BOOST"]              = {"PERSON": float(cb_blip_p), "ANIMAL": float(cb_blip_a), "VEHICLE": float(cb_blip_v)}
        cb_data["BBOX_SMALL_RATIO"]        = float(cb_bbox_ratio)
        cb_data["BBOX_SMALL_PERSON_BONUS"] = float(cb_bbox_bonus)
        save_json(cb_data, CLIP_BLIP_JSON)

        return "✅ All settings saved!"
    except Exception as e:
        return f"❌ Error: {str(e)}\n{traceback.format_exc()}"

# ==============================================================================
# LOG STREAMING
# ==============================================================================

class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        try:
            self.log_queue.put_nowait(self.format(record))
        except Exception:
            pass

def _run_scanner(input_path, output_path, mode, model_name,
                 use_refine, engine, use_fallback, is_clean_check,
                 no_sort, test_mode, device):
    """Start Scanner in thread and return (thread, result_holder, log_queue, q_handler)."""
    # Aggiorna model e device nel settings.json
    try:
        with open(SETTINGS_JSON, "r", encoding="utf-8") as f:
            settings_dict = json.load(f)
        settings_dict.setdefault("yolo_settings", {})
        settings_dict["yolo_settings"]["model_path"] = model_name.replace(".pt", "")
        settings_dict["yolo_settings"]["device"]     = device or "cuda"
        with open(SETTINGS_JSON, "w", encoding="utf-8") as f:
            json.dump(settings_dict, f, indent=4, ensure_ascii=False)
    except Exception as e:
        log.warning(f"Error updating settings.json: {e}")

    log_queue   = queue.Queue()
    q_handler   = QueueHandler(log_queue)
    q_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s : %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(q_handler)

    result_holder = {"error": None, "done": False, "scanner": None, "start_time": time.time()}

    def _run():
        try:
            get_logger(debug=test_mode)
            scanner = Scanner(
                mode=mode, device=device or None,
                is_refine=use_refine, is_fallback=use_fallback,
                is_test=test_mode, engine=engine,
                is_check_clean=is_clean_check, is_sort=not no_sort,
            )
            result_holder["scanner"] = scanner
            scanner.scan_folder(input_path, output_path)
        except Exception as e:
            result_holder["error"] = f"{str(e)}\n{traceback.format_exc()}"
        finally:
            result_holder["done"] = True

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread, result_holder, log_queue, q_handler


def _stream_logs(thread, result_holder, log_queue, q_handler, header=""):
    """Generator that streams logs to the UI."""
    log_lines = [header] if header else []
    while not result_holder["done"] or not log_queue.empty():
        try:
            line = log_queue.get(timeout=0.1)
            log_lines.append(line)
        except queue.Empty:
            pass
        yield "\n".join(log_lines[-80:])
        time.sleep(0.1)

    logging.getLogger().removeHandler(q_handler)
    
    scanner = result_holder.get("scanner")
    summary = ""
    if scanner and hasattr(scanner, "_get_final_summary"):
        try:
            elapsed = time.time() - result_holder.get("start_time", time.time())
            summary = "\n" + "─" * 50 + "\n" + scanner._get_final_summary(elapsed) + "\n" + "─" * 50
        except Exception:
            pass

    if result_holder["error"]:
        yield "\n".join(log_lines) + f"\n\n💥 ERROR:\n{result_holder['error']}"
    else:
        yield "\n".join(log_lines) + summary + "\n\n✅ COMPLETED!!"


def run_process(input_path, output_path, mode, model_name,
                use_refine, engine, use_fallback, is_clean_check, device):
    if not input_path:
        yield "⚠️ Specify the input folder."
        return
    input_p = Path(input_path)
    if not input_p.exists() or not input_p.is_dir():
        yield f"❌ Path '{input_path}' does not exist or is not a folder."
        return
    if is_clean_check and engine != "vision":
        yield "⚠️ Check Lens Health requires 'vision' engine."
        return

    final_output = output_path.strip() if output_path and output_path.strip() else input_path
    header = (f"🚀 Starting Scanner...\n📂 {input_path}\n"
              f"⚙️  Mode={mode} | Engine={'--refine --' + engine if use_refine else 'yolo only'} "
              f"| Fallback={use_fallback}\n{'─'*60}")

    thread, result_holder, log_queue, q_handler = _run_scanner(
        input_path, final_output, mode, model_name,
        use_refine, engine, use_fallback, is_clean_check,
        False, False, device 
    )
    yield from _stream_logs(thread, result_holder, log_queue, q_handler, header)


def run_test_process(input_path, output_path, mode, model_name,
                     use_refine, engine, use_fallback, no_sort, test_mode, device,
                     stride_sec, warmup_sec, stride_fast_sec, pre_roll_sec,num_occ,time_gap):
    if not input_path:
        yield "⚠️ Specify the input folder."
        return
    input_p = Path(input_path)
    if not input_p.exists() or not input_p.is_dir():
        yield f"❌ Path not valid."
        return

    final_output = output_path.strip() if output_path and output_path.strip() else input_path

    # Backup e applica valori di test
    backup_settings()
    try:
        with open(SETTINGS_JSON, "r", encoding="utf-8") as f:
            s = json.load(f)
        s["yolo_settings"]["vid_stride_sec"] = float(stride_sec)
        s["yolo_settings"]["dynamic_stride_settings"]["warmup_sec"]      = int(warmup_sec)
        s["yolo_settings"]["dynamic_stride_settings"]["stride_fast_sec"] = float(stride_fast_sec)
        s["yolo_settings"]["dynamic_stride_settings"]["pre_roll_sec"]    = int(pre_roll_sec)
        s["yolo_settings"]["num_occurrence"] = int(num_occ)
        s["yolo_settings"]["time_gap_sec"]   = int(time_gap)
        with open(SETTINGS_JSON, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=4, ensure_ascii=False)
    except Exception as e:
        yield f"❌ Error applying test parameters: {e}"
        return

    header = (f"🧪 TEST Scan\n📂 {input_path}\n"
              f"⚙️  stride={stride_sec}s | warmup={warmup_sec}s | fast={stride_fast_sec}s | preroll={pre_roll_sec}s\n"
              f"   Mode={mode} | Engine={'--refine --' + engine if use_refine else 'yolo only'} | NoSort={no_sort}\n{'─'*60}")

    thread, result_holder, log_queue, q_handler = _run_scanner(
        input_path, final_output, mode, model_name,
        use_refine, engine, use_fallback, False,
        no_sort, test_mode, device
    )
    yield from _stream_logs(thread, result_holder, log_queue, q_handler, header)


# ==============================================================================
# REAL-TIME SORTER
# ==============================================================================

_rt_thread     = None
_rt_stop_event = threading.Event()


def run_realtime(input_path, output_path, mode, model_name, engine, device, interval):
    global _rt_thread, _rt_stop_event

    if _rt_thread and _rt_thread.is_alive():
        yield "⚠️ Real-time already running. Stop it before restarting."
        return
    if not input_path:
        yield "⚠️ Specify the input folder."
        return
    input_p = Path(input_path)
    if not input_p.exists() or not input_p.is_dir():
        yield f"❌ Path not valid: {input_path}"
        return

    final_output = output_path.strip() if output_path and output_path.strip() else input_path
    _rt_stop_event.clear()

    log_queue = queue.Queue()
    q_handler = QueueHandler(log_queue)
    q_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s : %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(q_handler)

    def _loop():
        cycle = 0
        while not _rt_stop_event.is_set():
            cycle += 1
            log.info(f"─── Cycle #{cycle} ───")
            try:
                get_logger(debug=False)
                with open(SETTINGS_JSON, "r", encoding="utf-8") as f:
                    s = json.load(f)
                s.setdefault("yolo_settings", {})
                s["yolo_settings"]["model_path"] = model_name.replace(".pt", "")
                s["yolo_settings"]["device"]     = device or "cuda"
                with open(SETTINGS_JSON, "w", encoding="utf-8") as f:
                    json.dump(s, f, indent=4, ensure_ascii=False)

                scanner = Scanner(
                    mode=mode, device=device or None,
                    is_refine=True, is_fallback=False,
                    is_test=False, engine=engine,
                    is_check_clean=False, is_sort=True,
                )
                scanner.scan_folder(input_path, final_output)
            except RuntimeError as e:
                log.critical(f"Errore: {e}")
                break
            except Exception as e:
                log.error(f"Error cycle #{cycle}: {e}")

            if not _rt_stop_event.is_set():
                log.info(f"Cycle #{cycle} completated. Next in {interval}s")
                _rt_stop_event.wait(timeout=interval)

        log.info("Real-time sorter stopped.")

    _rt_thread = threading.Thread(target=_loop, daemon=True)
    _rt_thread.start()

    log_lines = []
    while _rt_thread.is_alive() or not log_queue.empty():
        try:
            line = log_queue.get(timeout=0.1)
            log_lines.append(line)
        except queue.Empty:
            pass
        yield "\n".join(log_lines[-80:])
        time.sleep(0.1)

    logging.getLogger().removeHandler(q_handler)
    yield "\n".join(log_lines) + "\n\n🛑 Real-time stopped."


def stop_realtime():
    global _rt_stop_event
    _rt_stop_event.set()
    return "🛑 Stop requested — the current cycle will finish and then stop."


# ==============================================================================
# TOOLS: GROUND TRUTH & COMPARE
# ==============================================================================

def generate_gt(input_dir, output_dir, check_dupl):
    from smart_surveillance_sorter.generate_ground_truth import genera_ground_truth, check_duplicates_with_log
    from smart_surveillance_sorter.constants import GROUND_TRUTH
    if not input_dir:
        return "⚠️ Specify the input folder."
    input_p = Path(input_dir)
    if not input_p.exists():
        return f"❌ Folder not found: {input_dir}"
    out_p = Path(output_dir.strip()) if output_dir and output_dir.strip() else input_p
    try:
        risultati = genera_ground_truth(str(input_p), log)
        out_file  = out_p / GROUND_TRUTH
        save_json(risultati, out_file)
        msg = f"✅ Ground Truth generated: {out_file}\n📹 Video found: {len(risultati)}"
        if check_dupl:
            import io, logging as _logging
            buf = io.StringIO()
            h   = _logging.StreamHandler(buf)
            log.addHandler(h)
            check_duplicates_with_log(str(input_p), log)
            log.removeHandler(h)
            dup_out = buf.getvalue()
            msg += f"\n\n⚠️ Duplicates:\n{dup_out}" if dup_out.strip() else "\n✅ No duplicates found."
        return msg
    except Exception as e:
        return f"❌ Error: {str(e)}"


def run_compare(session_dir, gt_file, res_file):
    from smart_surveillance_sorter.compare_results import compare_results
    lines = []
    class _ListLogger:
        def info(self, m):  lines.append(m)
        def error(self, m): lines.append(f"❌ {m}")
    try:
        compare_results(
            session_dir=session_dir.strip() if session_dir and session_dir.strip() else None,
            gt_file=gt_file.strip()   if gt_file  and gt_file.strip()  else None,
            res_file=res_file.strip() if res_file and res_file.strip() else None,
            log=_ListLogger()
        )
    except Exception as e:
        lines.append(f"❌ Errore: {str(e)}")
    return "\n".join(lines)


# ==============================================================================
# UI
# ==============================================================================

init_set, init_cam, init_prompt = load_configs()
current_settings = load_json(SETTINGS_JSON)
cb_set           = load_json(CLIP_BLIP_JSON)
current_prompts  = load_json(PROMPTS_JSON)
current_cameras  = load_json(CAMERAS_JSON)
camera_ids       = list(current_cameras.keys())
dss              = current_settings["yolo_settings"]["dynamic_stride_settings"]

with gr.Blocks(title="Smart Surveillance Sorter", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛡️ Smart Surveillance Sorter")

    with gr.Tabs():

        # ── TAB 1: RUN ────────────────────────────────────────────────────────
        with gr.TabItem("🚀 Run Scan"):
            with gr.Row():
                with gr.Column(scale=1):
                    run_input  = gr.Textbox(label="Input Directory", placeholder="/home/user/videos")
                    run_output = gr.Textbox(label="Output Directory (optional)")
                    run_mode   = gr.Radio(["full", "person", "person_animal"], label="Mode", value="full")

                with gr.Column(scale=1):
                    run_model  = gr.Dropdown(choices=get_available_models(), label="YOLO Model", value="yolov8l.pt")
                    run_device = gr.Radio(choices=["cuda", "cpu", "mps"], label="Device", value="cuda")

                    with gr.Group():
                        gr.Markdown("### 🧠 Engines")
                        with gr.Row():
                            run_refine = gr.Checkbox(label="✨ Enable Refine", value=True)
                            run_engine = gr.Radio(choices=["blip", "vision"], label="Engine", value="blip")
                        with gr.Row():
                            run_fallback    = gr.Checkbox(label="🔍 Fallback", value=False,
                                                          info="blip→vision on discordant | vision→NVR imgs low conf.")
                            run_check_clean = gr.Checkbox(label="🕸️ Check Lens", value=False,
                                                          info="Only with vision engine")
                        run_engine.change(fn=update_engines_availability, inputs=run_engine,
                                          outputs=[run_fallback, run_check_clean])
                        gr.Markdown("_`blip`: fast. `vision`: Ollama/Qwen (slow)._")

            run_btn    = gr.Button("▶ START SCAN", variant="primary", size="lg")
            run_status = gr.Markdown("")
            run_log    = gr.Textbox(label="📋 Check the terminal for detailed logs",
                                    interactive=False, lines=8)

            # run_btn.click(
            #     fn=disable_btns,
            #     outputs=ALL_BTNS,
            #     queue=False,
            # ).then(
            #     fn=run_process,
            #     inputs=[run_input, run_output, run_mode, run_model,
            #             run_refine, run_engine, run_fallback, run_check_clean, run_device],
            #     outputs=run_log,
            #     queue=True,
            # ).then(
            #     fn=enable_btns,
            #     outputs=ALL_BTNS,
            #     queue=False,
            # )

        # ── TAB 2: REAL-TIME ──────────────────────────────────────────────────
        with gr.TabItem("🔄 Real-Time"):
            gr.Markdown("_Continuous loop — scans new videos every N seconds._")
            with gr.Row():
                with gr.Column(scale=1):
                    rt_input    = gr.Textbox(label="Input Directory", placeholder="/home/user/videos")
                    rt_output   = gr.Textbox(label="Output Directory (optional)")
                    rt_mode     = gr.Radio(["full", "person", "person_animal"], label="Modalità", value="person")
                    rt_interval = gr.Slider(10, 300, step=10, value=60, label="Interval between cycles (sec)")

                with gr.Column(scale=1):
                    rt_model  = gr.Dropdown(choices=get_available_models(), label="YOLO Model", value="yolov8l.pt")
                    rt_device = gr.Radio(choices=["cuda", "cpu", "mps"], label="Device", value="cuda")
                    with gr.Group():
                        gr.Markdown("### 🧠 Engine")
                        rt_engine = gr.Radio(choices=["blip", "vision"], label="Engine", value="blip")
                        gr.Markdown("_Fallback and Test mode not available in real-time._")

            with gr.Row():
                rt_start_btn = gr.Button("▶ Start Real-Time", variant="primary")
                rt_stop_btn  = gr.Button("⏹ Stop", variant="stop")
            rt_status = gr.Markdown("")
            rt_log    = gr.Textbox(label="Log Real-Time", interactive=False, lines=12)

            # rt_start_btn.click(
            #     fn=disable_btns,
            #     outputs=ALL_BTNS,
            #     queue=False,
            # ).then(
            #     fn=run_realtime,
            #     inputs=[rt_input, rt_output, rt_mode, rt_model, rt_engine, rt_device, rt_interval],
            #     outputs=rt_log,
            #     queue=True,
            # ).then(
            #     fn=enable_btns,
            #     outputs=ALL_BTNS,
            #     queue=False,
            # )
            rt_stop_btn.click(
                fn=stop_realtime,
                outputs=rt_status,
                queue=False,
            ).then(
                fn=enable_btns,
                outputs=ALL_BTNS,
                queue=False,
            )

        # ── TAB 3: TEST ───────────────────────────────────────────────────────
        with gr.TabItem("🧪 Test & Tuning"):
            gr.Markdown("_Scan with temporary parameters. Settings are automatically restored after the test._")
            with gr.Row():
                with gr.Column(scale=1):
                    test_input  = gr.Textbox(label="Input Directory", placeholder="/home/user/videos")
                    test_output = gr.Textbox(label="Output Directory (optional)")
                    test_mode_r = gr.Radio(["full", "person", "person_animal"], label="Mode", value="full")
                    with gr.Row():
                        test_is_test = gr.Checkbox(label="🧪 Test mode (copy)", value=True)
                        test_no_sort = gr.Checkbox(label="🚫 No Sort",           value=True)

                with gr.Column(scale=1):
                    test_model  = gr.Dropdown(choices=get_available_models(), label="YOLO Model", value="yolov8l.pt")
                    test_device = gr.Radio(choices=["cuda", "cpu", "mps"], label="Device", value="cuda")
                    with gr.Group():
                        gr.Markdown("### 🧠 Engines")
                        with gr.Row():
                            test_refine  = gr.Checkbox(label="✨ Enable Refine", value=True)
                            test_engine  = gr.Radio(choices=["blip", "vision"], label="Engine", value="blip")
                        test_fallback = gr.Checkbox(label="🔍 Fallback", value=False)

            gr.Markdown("### 🎚️ YOLO parameters (temporary for this test)")
            with gr.Row():
                test_stride_sec      = gr.Slider(0.1, 2.0, step=0.1, value=current_settings["yolo_settings"].get("vid_stride_sec", 0.6),        label="Stride (sec)")
                test_warmup_sec      = gr.Slider(1, 15,   step=1,   value=dss.get("warmup_sec", 5),          label="Warmup (sec)")
                test_stride_fast_sec = gr.Slider(0.5, 3.0, step=0.1, value=dss.get("stride_fast_sec", 1.0),  label="Stride Fast (sec)")
                test_pre_roll_sec    = gr.Slider(5, 30,   step=1,   value=dss.get("pre_roll_sec", 20),       label="Pre Roll (sec)")
            with gr.Row():
                test_num_occ = gr.Slider(1, 10, step=1, value=current_settings["yolo_settings"]["num_occurrence"], label="Num. occurrences")
                test_time_gap = gr.Slider(1, 10, step=1, value=current_settings["yolo_settings"]["time_gap_sec"],   label="Time Gap (sec)")

            with gr.Row():
                test_btn   = gr.Button("▶ Start Test Scan", variant="primary", size="lg")
                restore_btn = gr.Button("↩️ Restore Backup", variant="secondary")

            test_status = gr.Markdown("_The backup is created automatically before each test scan._")
            test_log    = gr.Textbox(label="Log Test", interactive=False, lines=12)

            # test_btn.click(
            #     fn=disable_btns,
            #     outputs=ALL_BTNS,
            #     queue=False,
            # ).then(
            #     fn=run_test_process,
            #     inputs=[test_input, test_output, test_mode_r, test_model,
            #             test_refine, test_engine, test_fallback, test_no_sort, test_is_test, test_device,
            #             test_stride_sec, test_warmup_sec, test_stride_fast_sec, test_pre_roll_sec,
            #             test_num_occ, test_time_gap],
            #     outputs=test_log,
            #     queue=True,
            # ).then(
            #     fn=enable_btns,
            #     outputs=ALL_BTNS,
            #     queue=False,
            # )
            
            restore_btn.click(
                fn=restore_settings_backup,
                outputs=[test_status, test_stride_sec, test_warmup_sec, test_stride_fast_sec, test_pre_roll_sec]
            )

        # ── TAB 4: TOOLS ──────────────────────────────────────────────────────
        with gr.TabItem("📊 Tools"):
            with gr.Tabs():

                with gr.Tab("📋 Ground Truth"):
                    gr.Markdown("_Generate `ground_truth.json` by scanning subfolders (person/, animal/, vehicle/, others/)._")
                    with gr.Row():
                        gt_input  = gr.Textbox(label="Manually sorted video folder")
                        gt_output = gr.Textbox(label="Output directory (default=input)")
                    gt_dupl = gr.Checkbox(label="Check for Duplicates", value=True)
                    gt_btn  = gr.Button("📋 Generate Ground Truth", variant="primary")
                    gt_out  = gr.Textbox(label="Risultats", interactive=False, lines=6)
                    gt_btn.click(fn=generate_gt, inputs=[gt_input, gt_output, gt_dupl], outputs=gt_out)

                with gr.Tab("📊 Compare Results"):
                    gr.Markdown("_Compare `ground_truth.json` with `classification_results.json`._")
                    with gr.Accordion("📁 Use directory (search for files automatically)", open=True):
                        cmp_dir = gr.Textbox(label="Directory session")
                    with gr.Accordion("📄 Or specify the files manually", open=False):
                        with gr.Row():
                            cmp_gt  = gr.Textbox(label="Path ground_truth.json")
                            cmp_res = gr.Textbox(label="Path classification_results.json")
                    cmp_btn = gr.Button("📊 Compare", variant="primary")
                    cmp_out = gr.Textbox(label="Risultats", interactive=False, lines=20)
                    cmp_btn.click(fn=run_compare, inputs=[cmp_dir, cmp_gt, cmp_res], outputs=cmp_out)

        # ── TAB 5: SETTINGS ───────────────────────────────────────────────────
        with gr.TabItem("⚙️ Settings"):
            with gr.Tabs():

                with gr.Tab("🛠️ General"):
                    with gr.Accordion("📂 General", open=True):
                        with gr.Row():
                            city        = gr.Textbox(label="City", value=current_settings.get("city", ""))
                            priority    = gr.Textbox(label="Priority Hierarchy",
                                                     value=", ".join(current_settings["classification_settings"]["priority_hierarchy"]))
                            save_others = gr.Checkbox(label="Save 'Others'",
                                                      value=current_settings["classification_settings"]["save_others"])
                        with gr.Row():
                            fn_template = gr.Textbox(label="Filename Template", value=current_settings["storage_settings"]["filename_template"])
                            ts_format   = gr.Textbox(label="Timestamp Format",  value=current_settings["storage_settings"]["timestamp_format"])
                            struct_type = gr.Dropdown(choices=["camera_first", "date_first"], label="Folder Structure",
                                                      value=current_settings["storage_settings"]["structure_type"])

                    with gr.Accordion("🤖 YOLO", open=False):
                        with gr.Row():
                            y_mod = gr.Textbox(label="Model Name", value=current_settings["yolo_settings"]["model_path"])
                            y_dev = gr.Radio(choices=["cuda", "cpu", "mps"], label="Device",
                                             value=current_settings["yolo_settings"]["device"])
                        with gr.Row():
                            y_stride_sec     = gr.Number(label="Video Stride (sec)", value=current_settings["yolo_settings"].get("vid_stride_sec", 0.6))
                            y_occ            = gr.Slider(1, 10, step=1, label="Num. Occorrenze", value=current_settings["yolo_settings"]["num_occurrence"])
                            y_gap            = gr.Number(label="Time Gap (sec)", value=current_settings["yolo_settings"]["time_gap_sec"])
                        gr.Markdown("**Dynamic Stride**")
                        with gr.Row():
                            warmup_sec       = gr.Slider(1, 15,   step=1,   value=dss.get("warmup_sec", 5),         label="Warmup (sec)")
                            stride_fast_sec  = gr.Slider(0.5, 3.0, step=0.1, value=dss.get("stride_fast_sec", 1.0), label="Stride Fast (sec)")
                            pre_roll_sec     = gr.Slider(5, 30,   step=1,   value=dss.get("pre_roll_sec", 20),      label="Pre Roll (sec)")
                            cd_sec           = gr.Slider(1, 20,   step=1,   value=dss.get("cooldown_sec", 5),       label="Cooldown (sec)")

                    with gr.Accordion("👁️ Vision AI (Ollama)", open=False):
                        with gr.Row():
                            v_mod  = gr.Textbox(label="Model", value=current_settings["vision_settings"]["model_name"])
                            v_temp = gr.Slider(0, 2, value=current_settings["vision_settings"]["temperature"], label="Temperature")
                        with gr.Row():
                            ollama_ip       = gr.Textbox(label="IP",   value=current_settings["vision_settings"]["ollama_conf"]["ip"])
                            ollama_port     = gr.Textbox(label="Port", value=str(current_settings["vision_settings"]["ollama_conf"]["port"]))
                            validate_btn    = gr.Button("🔍 Check Ollama", variant="secondary")
                            validate_status = gr.Markdown("_Not validated_")
                        with gr.Row():
                            v_topk       = gr.Number(label="Top K",       value=current_settings["vision_settings"]["top_k"])
                            v_topp       = gr.Slider(0, 1, value=current_settings["vision_settings"]["top_p"], label="Top P")
                            v_num_predict = gr.Number(label="Num Predict", value=current_settings["vision_settings"].get("num_predict", 1024))

                    with gr.Accordion("🎛️ CLIP+BLIP Engine", open=False):
                        gr.Markdown("_Global — overridden on a per-camera basis in the Cameras tab._")
                        with gr.Row():
                            cb_weight_crop  = gr.Slider(0, 1, step=0.05, value=cb_set.get("FINAL_WEIGHT_CROP", 0.7),  label="Weight Crop")
                            cb_weight_frame = gr.Slider(0, 1, step=0.05, value=cb_set.get("FINAL_WEIGHT_FRAME", 0.3), label="Weight Frame")
                            cb_night_boost  = gr.Slider(0, 1, step=0.05, value=cb_set.get("YOLO_NIGHT_BOOST", 0.3),   label="Night Boost PERSON")
                        gr.Markdown("**THRESHOLD**")
                        with gr.Row():
                            cb_thr_p = gr.Slider(0, 1, step=0.01, value=cb_set["THRESHOLD"]["PERSON"],  label="PERSON")
                            cb_thr_v = gr.Slider(0, 1, step=0.01, value=cb_set["THRESHOLD"]["VEHICLE"], label="VEHICLE")
                            cb_thr_a = gr.Slider(0, 1, step=0.01, value=cb_set["THRESHOLD"]["ANIMAL"],  label="ANIMAL")
                        gr.Markdown("**FAKE PENALTY WEIGHT**")
                        with gr.Row():
                            cb_fpw_p = gr.Slider(0, 1, step=0.05, value=cb_set["FAKE_PENALTY_WEIGHT"]["PERSON"],  label="PERSON")
                            cb_fpw_a = gr.Slider(0, 1, step=0.05, value=cb_set["FAKE_PENALTY_WEIGHT"]["ANIMAL"],  label="ANIMAL")
                            cb_fpw_v = gr.Slider(0, 1, step=0.05, value=cb_set["FAKE_PENALTY_WEIGHT"]["VEHICLE"], label="VEHICLE")
                        gr.Markdown("**BLIP BOOST**")
                        with gr.Row():
                            cb_blip_p = gr.Slider(0, 1, step=0.05, value=cb_set["BLIP_BOOST"]["PERSON"],  label="PERSON")
                            cb_blip_a = gr.Slider(0, 1, step=0.05, value=cb_set["BLIP_BOOST"]["ANIMAL"],  label="ANIMAL")
                            cb_blip_v = gr.Slider(0, 1, step=0.05, value=cb_set["BLIP_BOOST"]["VEHICLE"], label="VEHICLE")
                        gr.Markdown("**BBOX Small Bonus**")
                        with gr.Row():
                            cb_bbox_ratio = gr.Slider(0, 0.2, step=0.005, value=cb_set.get("BBOX_SMALL_RATIO", 0.04),        label="Ratio Soglia")
                            cb_bbox_bonus = gr.Slider(0, 0.5, step=0.05,  value=cb_set.get("BBOX_SMALL_PERSON_BONUS", 0.15), label="Bonus PERSON")

                    with gr.Accordion("⚖️ Scoring System", open=False):
                        with gr.Row():
                            w_high = gr.Number(label="Weight High", value=current_settings["scoring_system"]["weights"]["score_high"])
                            w_mid  = gr.Number(label="Weight Mid",  value=current_settings["scoring_system"]["weights"]["score_mid"])
                            w_low  = gr.Number(label="Weight Low",  value=current_settings["scoring_system"]["weights"]["score_low"])
                        with gr.Row():
                            sc_p = gr.Number(label="Min Person",  value=current_settings["scoring_system"]["thresholds"]["person"])
                            sc_a = gr.Number(label="Min Animal",  value=current_settings["scoring_system"]["thresholds"]["animal"])
                            sc_v = gr.Number(label="Min Vehicle", value=current_settings["scoring_system"]["thresholds"]["vehicle"])
                        gr.Markdown("**YOLO Override**")
                        with gr.Row():
                            ov_min_conf  = gr.Slider(0.3, 0.9, step=0.01, value=current_settings["scoring_system"].get("yolo_override", {}).get("person_min_conf", 0.58),               label="Person Min Conf")
                            ov_min_score = gr.Slider(0.5, 5.0, step=0.1,  value=current_settings["scoring_system"].get("yolo_override", {}).get("min_total_score_to_skip_override", 1.2), label="Min Score Skip Override")

                    with gr.Row():
                        save_all_btn      = gr.Button("💾 SAVE ALL SETTINGS", variant="primary", size="lg")
                        restore_def_btn   = gr.Button("🔄 Restore Default", variant="secondary")
                    status_save = gr.Markdown("")

                    validate_btn.click(fn=ui_validate_ollama, inputs=[ollama_ip, ollama_port, v_mod], outputs=validate_status)
                    save_all_btn.click(
                        fn=save_comprehensive_settings,
                        inputs=[
                            city, priority, save_others, fn_template, ts_format, struct_type,
                            y_mod, y_dev, y_stride_sec, y_occ, y_gap,
                            warmup_sec, stride_fast_sec, pre_roll_sec, cd_sec,
                            v_mod, v_temp, ollama_ip, ollama_port, v_topk, v_topp, v_num_predict,
                            w_high, w_mid, w_low, sc_p, sc_a, sc_v, ov_min_conf, ov_min_score,
                            cb_weight_crop, cb_weight_frame, cb_night_boost,
                            cb_thr_p, cb_thr_v, cb_thr_a,
                            cb_fpw_p, cb_fpw_a, cb_fpw_v,
                            cb_blip_p, cb_blip_a, cb_blip_v,
                            cb_bbox_ratio, cb_bbox_bonus,
                        ],
                        outputs=status_save,
                    )
                    restore_def_btn.click(fn=restore_settings_default, outputs=status_save)

                # ── Cameras ───────────────────────────────────────────────────
                with gr.Tab("📹 Cameras"):
                    with gr.Row():
                        cam_selector = gr.Dropdown(choices=camera_ids, label="Telecamera",
                                                   value=camera_ids[0] if camera_ids else None, interactive=True)
                        add_cam_btn  = gr.Button("➕ Add", variant="secondary")
                    with gr.Group():
                        gr.Markdown("### 📝 Settings")
                        with gr.Row():
                            c_name = gr.Textbox(label="Nome")
                            c_loc  = gr.Textbox(label="Location")
                        with gr.Row():
                            c_patterns = gr.Textbox(label="Search Patterns (virgola)", placeholder="_00_, ch00")
                            c_priority = gr.Dropdown(choices=["person", "animal", "vehicle"], label="Priority")
                        c_desc = gr.Textbox(label="Description (used in Vision AI prompt)", lines=2)
                        with gr.Row():
                            c_dynamic = gr.Checkbox(label="Dynamic Stride")
                            c_ignore  = gr.Textbox(label="Labels to ignore (COCO, comma)")
                        gr.Markdown("### 🎯 YOLO Thresholds")
                        gr.Markdown("**Day**")
                        with gr.Row():
                            cth_p = gr.Slider(0, 1, step=0.01, label="Person")
                            cth_v = gr.Slider(0, 1, step=0.01, label="Vehicle")
                            cth_a = gr.Slider(0, 1, step=0.01, label="Animal")
                        gr.Markdown("**Night** — -1 = usa day thresholds")
                        with gr.Row():
                            cth_np = gr.Slider(-1, 1, step=0.01, value=-1.0, label="Person night")
                            cth_nv = gr.Slider(-1, 1, step=0.01, value=-1.0, label="Vehicle night")
                            cth_na = gr.Slider(-1, 1, step=0.01, value=-1.0, label="Animal night")
                        gr.Markdown("### 🎛️ CLIP+BLIP Override  (-1 = use global)")
                        gr.Markdown("**Fake Weights**")
                        with gr.Row():
                            fw_ground = gr.Slider(0, 1, step=0.05, label="GROUND")
                            fw_garden = gr.Slider(0, 1, step=0.05, label="GARDEN")
                            fw_shoe   = gr.Slider(0, 1, step=0.05, label="SHOE")
                            fw_wood   = gr.Slider(0, 1, step=0.05, label="WOOD")
                        gr.Markdown("**THRESHOLD override**")
                        with gr.Row():
                            cam_thr_p = gr.Slider(-1, 1, step=0.01, value=-1.0, label="PERSON")
                            cam_thr_v = gr.Slider(-1, 1, step=0.01, value=-1.0, label="VEHICLE")
                            cam_thr_a = gr.Slider(-1, 1, step=0.01, value=-1.0, label="ANIMAL")
                        gr.Markdown("**FAKE PENALTY WEIGHT override**")
                        with gr.Row():
                            cam_fpw_p = gr.Slider(-1, 1, step=0.05, value=-1.0, label="PERSON")
                            cam_fpw_a = gr.Slider(-1, 1, step=0.05, value=-1.0, label="ANIMAL")
                            cam_fpw_v = gr.Slider(-1, 1, step=0.05, value=-1.0, label="VEHICLE")

                    cam_inputs  = [cam_selector, c_name, c_loc, c_patterns, c_priority, c_desc, c_dynamic, c_ignore,
                                   cth_p, cth_v, cth_a, cth_np, cth_nv, cth_na,
                                   fw_ground, fw_garden, fw_shoe, fw_wood,
                                   cam_thr_p, cam_thr_v, cam_thr_a,
                                   cam_fpw_p, cam_fpw_a, cam_fpw_v]
                    cam_outputs = [c_name, c_loc, c_patterns, c_priority, c_desc, c_dynamic, c_ignore,
                                   cth_p, cth_v, cth_a, cth_np, cth_nv, cth_na,
                                   fw_ground, fw_garden, fw_shoe, fw_wood,
                                   cam_thr_p, cam_thr_v, cam_thr_a,
                                   cam_fpw_p, cam_fpw_a, cam_fpw_v]

                    with gr.Row():
                        save_cam_btn   = gr.Button("💾 Save Camera", variant="primary")
                        delete_cam_btn = gr.Button("🗑️ Delete", variant="stop")
                    status_cam = gr.Markdown("")

                    save_cam_btn.click(fn=save_single_camera,   inputs=cam_inputs,   outputs=status_cam)
                    delete_cam_btn.click(fn=delete_camera,      inputs=cam_selector, outputs=[cam_selector, status_cam])
                    add_cam_btn.click(fn=add_new_camera,         outputs=cam_selector)
                    cam_selector.change(fn=load_camera_details,  inputs=cam_selector, outputs=cam_outputs)
                    demo.load(fn=load_camera_details,             inputs=cam_selector, outputs=cam_outputs)

                # ── Prompt Editor ─────────────────────────────────────────────
                with gr.Tab("📝 Prompt AI"):
                    with gr.Accordion("📢 System Instruction", open=True):
                        p_sys   = gr.Textbox(label="System Instruction", value=current_prompts["shared_components"]["system_instruction"])
                        p_rules = gr.Textbox(label="Mandatory Rules",    value=current_prompts["shared_components"]["mandatory_rules"], lines=3)
                    with gr.Accordion("🎯 Class Descriptions", open=True):
                        desc_p     = gr.Textbox(label="PERSON",          value=current_prompts["class_descriptions"]["PERSON"],      lines=2)
                        desc_a     = gr.Textbox(label="ANIMAL",          value=current_prompts["class_descriptions"]["ANIMAL"],      lines=2)
                        desc_v     = gr.Textbox(label="VEHICLE",         value=current_prompts["class_descriptions"]["VEHICLE"],     lines=2)
                        desc_clean = gr.Textbox(label="LENS CLEAN CHECK",value=current_prompts["class_descriptions"]["CLEAN_CHECK"], lines=3)
                    with gr.Accordion("🧩 Modules / Headers (Advanced)", open=False):
                        m_crop  = gr.Textbox(label="Mission Crop Header", value=current_prompts["modules"]["analyst_mission_crop"], lines=3)
                        m_fall  = gr.Textbox(label="Fallback Header",     value=current_prompts["modules"]["fallback_header"],      lines=2)
                        m_clean = gr.Textbox(label="Clean Check Header",  value=current_prompts["modules"]["clean_header"],         lines=2)
                        with gr.Accordion("🔍 Dynamic Prompt Preview", open=False):
                            with gr.Row():
                                test_mode_p   = gr.Dropdown(choices=["full", "person", "person_animal", "clean_check"], label="Mode", value="full")
                                test_has_crop = gr.Checkbox(label="Simulate Crop")
                                test_is_fb    = gr.Checkbox(label="Simulate Fallback")
                            preview_btn    = gr.Button("🔨 Generate Preview", variant="secondary")
                            prompt_preview = gr.Code(label="Prompt finale", language="markdown", lines=15)
                    save_prompt_btn = gr.Button("💾 Save Prompt AI", variant="primary")
                    status_prompt   = gr.Markdown("")
                    preview_btn.click(
                        fn=preview_prompt_logic,
                        inputs=[p_sys, p_rules, desc_p, desc_a, desc_v, m_crop, m_fall, test_mode_p, test_has_crop, test_is_fb],
                        outputs=prompt_preview
                    )
                    save_prompt_btn.click(
                        fn=save_prompts_ui,
                        inputs=[p_sys, p_rules, desc_p, desc_a, desc_v, desc_clean, m_crop, m_fall, m_clean],
                        outputs=status_prompt
                    )

                # ── System ────────────────────────────────────────────────────
                with gr.Tab("🖥️ System"):
                    stop_btn   = gr.Button("🛑 Turn off WebUI", variant="stop")
                    status_sys = gr.Markdown("Status: Running")
                    stop_btn.click(fn=shutdown_server, outputs=status_sys)

    ALL_BTNS = [run_btn, test_btn, rt_start_btn]

    run_btn.click(
        fn=disable_btns,
        outputs=ALL_BTNS,
        queue=False,
    ).then(
        fn=run_process,
        inputs=[run_input, run_output, run_mode, run_model,
                run_refine, run_engine, run_fallback, run_check_clean, run_device],
        outputs=run_log,
        queue=True,
    ).then(
        fn=enable_btns,
        outputs=ALL_BTNS,
        queue=False,
    )

    test_btn.click(
        fn=disable_btns,
        outputs=ALL_BTNS,
        queue=False,
    ).then(
        fn=run_test_process,
        inputs=[test_input, test_output, test_mode_r, test_model,
                test_refine, test_engine, test_fallback, test_no_sort, test_is_test, test_device,
                test_stride_sec, test_warmup_sec, test_stride_fast_sec, test_pre_roll_sec,
                test_num_occ, test_time_gap],
        outputs=test_log,
        queue=True,
    ).then(
        fn=enable_btns,
        outputs=ALL_BTNS,
        queue=False,
    )

    rt_start_btn.click(
        fn=disable_btns,
        outputs=ALL_BTNS,
        queue=False,
    ).then(
        fn=run_realtime,
        inputs=[rt_input, rt_output, rt_mode, rt_model, rt_engine, rt_device, rt_interval],
        outputs=rt_log,
        queue=True,
    ).then(
        fn=enable_btns,
        outputs=ALL_BTNS,
        queue=False,
    )
                

import socket

def find_free_port(start=7860, end=7900):
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No free port found")

if __name__ == "__main__":
    port = find_free_port()

    demo.queue()
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        prevent_thread_lock=False,
    )