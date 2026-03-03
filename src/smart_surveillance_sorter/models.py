import logging
import torch
import os
from pathlib import Path
from ultralytics import YOLO
from smart_surveillance_sorter.constants import MODELS_DIR
log = logging.getLogger(__name__) 
def load_smart_yolo(model_name, device=None):
    """
    Load a YOLOv8 model by name, handling local loading or downloading.

    This helper reads the model file name from the settings, normalizes it
    to ensure a ``.pt`` extension, and resolves the full path using the
    project constant ``MODELS_DIR``. If the model file already exists
    locally, it is loaded directly; otherwise the function downloads
    the file from the Ultralytics hub, saves it to the local path,
    and returns the loaded model.

    Parameters
    ----------
    model_name : str
        Name of the YOLO model to load. The name may or may not include
        the ``.pt`` extension; the function normalizes it accordingly.
    device : torch.device or str, optional
        Device on which to load the model (e.g., ``"cpu"`` or ``"cuda"``).
        If not provided, the default device selection logic of the
        Ultralytics library is used.

    Returns
    -------
    ultralytics.YOLO
        The loaded YOLO model instance, ready for inference.

    Notes
    -----
    The function checks if the model file exists in ``MODELS_DIR``; if it
    does, it loads the local copy, otherwise it downloads the model
    from the Ultralytics hub, saves it locally, and then returns the
    instance. This approach ensures that the required model is always
    available for the smart surveillance sorter pipeline. [1]
    """

    # 2. Normalizza (assicura .pt) e usa la costante MODELS_DIR
    model_file = model_name if model_name.endswith(".pt") else f"{model_name}.pt"
    local_path = MODELS_DIR / model_file
    # 3. Caricamento fisico
    if local_path.exists():
        log.info(f"📦 [YOLO] Loading: {local_path}")
        model = YOLO(str(local_path))
    else:
        log.info(f"🌐 [YOLO] Download {model_file} in {MODELS_DIR}...")
        model = YOLO(str(MODELS_DIR / model_file))
    
    if device:
        target_device = device
    else:
        target_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    log.debug(f"[YOLO] → Utilizzo device: {target_device}")
    model.to(target_device)
    
    return model

