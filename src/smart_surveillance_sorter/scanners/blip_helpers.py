def build_vqa_question(category, detection_groups):
    """
    Costruisce la domanda per BLIP partendo dalla categoria del frame 
    e la mappatura dei gruppi.
    """
    # 1. Recupero etichette (es. ["dog", "cat"])
    # Facciamo l'upper per evitare problemi di case-sensitivity
    labels = detection_groups.get(category.upper(), [])
    
    if not labels:
        return "Is there any object in this image?"

    # 2. Formattazione naturale (car, truck or bus)
    if len(labels) > 1:
        labels_str = ", ".join(labels[:-1]) + " or " + labels[-1]
    else:
        labels_str = labels[0]

    # 3. Costruzione domanda specifica
    cat_upper = category.upper()
    if cat_upper == "ANIMAL":
        return f"Is there an animal like a {labels_str}?"
    elif cat_upper == "PERSON":
        return "Is there a person, a human figure or an animal in this image?"
    elif cat_upper == "VEHICLE":
        return f"Is there a vehicle like a {labels_str}?"
    
    return f"Is there a {labels_str}?"