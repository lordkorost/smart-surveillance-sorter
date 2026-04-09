import logging


LABEL_TO_CAT = {
        "person": "PERSON", "people": "PERSON",
        "dog": "ANIMAL", "cat": "ANIMAL", "animal": "ANIMAL",
        "car": "VEHICLE", "truck": "VEHICLE", "vehicle": "VEHICLE"
    }

log = logging.getLogger(__name__) 
def build_dynamic_prompt(prompts_config, cam_cfg, mode="full", has_crop=False, is_fallback=False):
    """
    Build a dynamically generated prompt for vision model based on configuration.
    
    Args:
        prompts_config: Prompt configuration dictionary
        cam_cfg: Camera configuration
        mode: Detection mode (full, person, person_animal)
        has_crop: Whether request includes cropped detection regions
        is_fallback: Whether using fallback mode
        
    Returns:
        Formatted prompt string for vision model
    """
    mode_map = {
        "full": ["PERSON", "ANIMAL", "VEHICLE"],
        "person": ["PERSON"],
        "person_animal": ["PERSON", "ANIMAL"]
    }
    
    # Classes allowed by command (e.g. --mode person -> only PERSON)
    allowed_by_mode = mode_map.get(mode, ["PERSON"]) 

    # 2. See what to ignore for THIS camera specifically
    ignore_labels = cam_cfg.get("filters", {}).get("ignore_labels", [])
    ignored_cats = {LABEL_TO_CAT[l] for l in ignore_labels if l in LABEL_TO_CAT}

    # 3. Final result: Mode classes - Ignored classes
    active_classes = [c for c in allowed_by_mode if c not in ignored_cats]

    # 4. Build textual hierarchy
    hierarchy_lines = []
    class_descriptions = prompts_config["class_descriptions"]
    
    for idx, cls in enumerate(active_classes, start=1):
        desc_text = class_descriptions.get(cls)
        hierarchy_lines.append(f"{idx}. {cls}: {desc_text}, output '{cls.lower()}'.")
    
    hierarchy_lines.append(f"{len(active_classes)+1}. NOTHING: If none of the above are present, output 'nothing'.")

    # 5. Fill the template
    shared = prompts_config.get("shared_components", {})
    modules = prompts_config.get("modules", {})

    context = {
        "system_instruction": shared.get("system_instruction", "You are an AI Analyst."),
        "desc": cam_cfg.get("desc", "Outdoor area"),
        "hierarchy": "\n".join(hierarchy_lines),
        "rules": shared.get("mandatory_rules", "Follow the output format strictly."),
        "allowed_outputs": " | ".join([c.lower() for c in active_classes] + ["nothing"])
    }

    # Get templates (default to empty dict to avoid crash)
    all_templates = prompts_config.get("templates", {})

    if is_fallback:
        template = all_templates.get("fallback")
        context["fallback_header"] = modules.get("fallback_header", "")
    elif has_crop:
        template = all_templates.get("with_crop")
        context["crop_header"] = modules.get("analyst_mission_crop", "")
    else:
        template = all_templates.get("standard")

    # 6. Handle error if template is None or empty string
    if not template:
        log.error(f"Missing template Fallback/Crop/Standard in prompts.json")
        raise ValueError("Prompt config incomplete: check prompts.json")
    

    return template.format(**context)

def build_clean_prompt(prompts_config, cam_cfg):
    """
    Build prompt for lens cleanliness analysis.
    
    Args:
        prompts_config: Prompt configuration dictionary
        cam_cfg: Camera configuration
        
    Returns:
        Formatted prompt for lens cleanliness assessment
    """
    shared = prompts_config.get("shared_components", {})
    modules = prompts_config.get("modules", {})
    class_descs = prompts_config.get("class_descriptions", {})
    templates = prompts_config.get("templates", {})

    # Prepare dictionary with real data
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