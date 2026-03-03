from datetime import datetime
from pathlib import Path

import cv2

from smart_surveillance_sorter.utils import get_crop_coordinates


def extract_frames_with_cache(
    cap,
    detections,
    fps,
    video_path,
    frames_dir,
    frames_per_category,
):
    """
    Extracts frames from a video at specified intervals.
    
    Parameters
    ----------
    video_path : Path
        Path to the video file to be analyzed.
    frame_rate : float, optional
        Number of frames to extract per second (default: 1.0).
    output_dir : Path | None, optional
        Directory where the extracted frames will be saved. If `None`, a temporary directory within the input directory is created.
    
    Returns
    -------
    list[Path]
        List of paths to the extracted frame files.
    
    Notes
    -----
    * The function is used by `Scanner` to prepare data for both `YoloEngine` and `VisionEngine` [5].
    * Frames are saved in PNG format with sequential naming.
    
    Examples
    --------
    >>> frames = extract_frames(Path("cam01.mp4"), frame_rate=2)
    >>> len(frames)
    120
    """
    
    saved_frames = []
    frame_cache = {}
    frames_dir = Path(frames_dir)
    # 1. Creiamo una lista piatta di tutti i frame unici che dobbiamo estrarre
    # Ordiniamo per categoria e confidenza 
    target_detections = []
    for cat, items in detections.items():
        items.sort(key=lambda x: x["confidence"], reverse=True)
        # Prendiamo solo i top N per categoria
        for i, det in enumerate(items[:frames_per_category]):
            det["cat_rank"] = i
            det["category"] = cat
            target_detections.append(det)

    # 2. Estrazione
    for det in target_detections:
        frame_idx = det["frame_idx"]
        cat = det["category"]
        rank = det["cat_rank"]

        # Recupero frame 
        if frame_idx not in frame_cache:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frame_cache[frame_idx] = frame
        else:
            frame = frame_cache[frame_idx]

        # -------- Calcolo Timestamp --------
        # Usiamo mtime del file + offset del frame
        video_start_ts = video_path.stat().st_mtime
        ts_sec = frame_idx / fps
        ts_iso = datetime.fromtimestamp(video_start_ts + ts_sec).isoformat()

        # -------- Salvataggio Frame Intero (PULITO per Vision AI) --------
        out_name = f"{video_path.stem}_{cat}_{rank}.jpg"
        out_path = frames_dir / out_name
        cv2.imwrite(str(out_path), frame)
        

        # -------- Salva Crop (Dettaglio per controllo) --------
        c_x1, c_y1, c_x2, c_y2 = get_crop_coordinates(det["bbox"], frame.shape)
        cropped_frame = frame[c_y1:c_y2, c_x1:c_x2]
        
        out_name_crop = f"{video_path.stem}_{cat}_{rank}_crop.jpg"
        out_path_crop = frames_dir / out_name_crop
        if cropped_frame.size > 0: # Evitiamo crash su crop vuoti
            cv2.imwrite(str(out_path_crop), cropped_frame)
        

        # -------- Record Unico per Frame + Crop --------
        saved_frames.append({
            "category": cat,
            "yolo_label": det.get("yolo_label", ""),
            "frame_path": str(out_path),
            "crop_path": str(out_path_crop),
            "confidence": det["confidence"],
            "yolo_reliable": det.get("yolo_reliable", False),
            "bbox": det["bbox"],
            "area_ratio": det.get("area_ratio", 0),
            "timestamp": ts_iso,
            "frame_idx": frame_idx
        })

    return saved_frames