import logging


LABEL_TO_CAT = {
        "person": "PERSON", "people": "PERSON",
        "dog": "ANIMAL", "cat": "ANIMAL", "animal": "ANIMAL",
        "car": "VEHICLE", "truck": "VEHICLE", "vehicle": "VEHICLE"
    }

log = logging.getLogger(__name__) 
def build_dynamic_prompt(prompts_config, cam_cfg, mode="full", has_crop=False, is_fallback=False):
    

        mode_map = {
            "full": ["PERSON", "ANIMAL", "VEHICLE"],
            "person": ["PERSON"],
            "person_animal": ["PERSON", "ANIMAL"]
        }
        
        # Classi permesse dal comando (es. --mode person -> solo PERSON)
        allowed_by_mode = mode_map.get(mode, ["PERSON"]) 

        # 2. Vediamo cosa ignorare per QUESTA telecamera specifica
        ignore_labels = cam_cfg.get("filters", {}).get("ignore_labels", [])
        ignored_cats = {LABEL_TO_CAT[l] for l in ignore_labels if l in LABEL_TO_CAT}

        # 3. Risultato finale: Classi del modo - Classi ignorate
        active_classes = [c for c in allowed_by_mode if c not in ignored_cats]

        # 4. Costruiamo la gerarchia testuale
        hierarchy_lines = []
        class_descriptions = prompts_config["class_descriptions"]
        
        for idx, cls in enumerate(active_classes, start=1):
            desc_text = class_descriptions.get(cls)
            hierarchy_lines.append(f"{idx}. {cls}: {desc_text}, output '{cls.lower()}'.")
        
        hierarchy_lines.append(f"{len(active_classes)+1}. NOTHING: If none of the above are present, output 'nothing'.")

        # 5. Riempiamo il template (Usiamo i .get per ogni componente del prompt)
        shared = prompts_config.get("shared_components", {})
        modules = prompts_config.get("modules", {})

        context = {
            "system_instruction": shared.get("system_instruction", "You are an AI Analyst."),
            "desc": cam_cfg.get("desc", "Outdoor area"),
            "hierarchy": "\n".join(hierarchy_lines),
            "rules": shared.get("mandatory_rules", "Follow the output format strictly."),
            "allowed_outputs": " | ".join([c.lower() for c in active_classes] + ["nothing"])
        }

        # Recuperiamo i template (Default a dizionario vuoto per evitare crash)
        all_templates = prompts_config.get("templates", {})

        if is_fallback:
            template = all_templates.get("fallback")
            context["fallback_header"] = modules.get("fallback_header", "")
        elif has_crop:
            template = all_templates.get("with_crop")
            context["crop_header"] = modules.get("analyst_mission_crop", "")
        else:
            template = all_templates.get("standard")

        # 6. Gestione errore se il template è None o stringa vuota
        if not template:
            log.error(f"Missing template Fallback/Crop/Standard in prompts.json")
            raise ValueError("Prompt config incomplete: check prompts.json")
        
        # # 5. Riempiamo il template
        # context = {
        #     "system_instruction": prompts_config["shared_components"]["system_instruction"],
        #     "desc": cam_cfg.get("desc", "Outdoor area"),
        #     "hierarchy": "\n".join(hierarchy_lines),
        #     "rules": prompts_config["shared_components"]["mandatory_rules"],
        #     "allowed_outputs": " | ".join([c.lower() for c in active_classes] + ["nothing"])
        # }

        # # Recuperiamo il dizionario dei template dalla configurazione passata
        # all_templates = prompts_config.get("templates", {})

        # if is_fallback:
        #     template = all_templates.get("fallback")
        #     context["fallback_header"] = prompts_config["modules"]["fallback_header"]
        # elif has_crop:
        #     template = all_templates.get("with_crop")
        #     context["crop_header"] = prompts_config["modules"]["analyst_mission_crop"]
        # else:
        #     # Qui usiamo il template standard
        #     template = all_templates.get("standard")

        # # Gestione errore se il template non esiste nel JSON
        # if not template:
        #     raise ValueError("Template non trovato nel file prompts.json")

        return template.format(**context)



def build_clean_prompt(prompts_config, cam_cfg):
    """
    Assembla il prompt per il controllo pulizia lenti leggendo tutto dal JSON.
    """
    shared = prompts_config.get("shared_components", {})
    modules = prompts_config.get("modules", {})
    class_descs = prompts_config.get("class_descriptions", {})
    templates = prompts_config.get("templates", {})

    # Prepariamo il dizionario con i dati reali
    context = {
        "clean_header": modules.get("clean_header"),
        "system_instruction": shared.get("system_instruction"),
        "desc": cam_cfg.get("desc", "N/A"),
        "class_desc": class_descs.get("CLEAN_CHECK"),
        "rules": shared.get("mandatory_rules")
    }

    template_str = templates.get("clean_check")
    
    # Formattazione pulita: iniettiamo i dati nel template del JSON
    return template_str.format(**context)