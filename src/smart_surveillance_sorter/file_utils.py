from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import re
import time

from smart_surveillance_sorter.file_sorter import FileSorter
from smart_surveillance_sorter.utils import save_json

log = logging.getLogger(__name__)

def build_index(input_dir,settings):
    """Build an index of files in the specified input directory.

    This function scans the input directory and creates a structured index 
    containing information about video and image files, organized by camera ID.
    It calculates statistics such as total videos, total images, and detects
    potential inconsistencies (e.g., fewer images than videos for a camera).

    Args:
        input_dir (str | Path): Directory path where the file index will be created.
        settings (dict): Configuration dictionary containing storage and processing settings.

    Returns:
        dict: A dictionary mapping camera_id lists of file metadata entries.

    Side Effects:
        Logs statistics about the number of videos, images, and cameras found.
        Warns if a camera has fewer images than videos detected.

    Example:
        >>> index = build_index("/path/to/input", settings_config)
        >>> print(f"Cameras found: {len(index)}")
    """
    log.info(f"Create index of files in folder={input_dir}...")
    index = {}
    # Retrieve settings from config
    storage_cfg = settings.get("storage_settings", {})
    template = storage_cfg.get("filename_template", "{nvr_name}_{camera_id}_{timestamp}")
    ts_format = storage_cfg.get("timestamp_format", "%Y%m%d%H%M%S")
    input_dir = Path(input_dir)
    extensions = {".mp4", ".mkv", ".avi", ".mov", ".jpg", ".jpeg"}
    file_list = [f for f in input_dir.iterdir() if f.suffix.lower() in extensions]

    for f in file_list:
        # --- WRITE PROTECTION FOR FILE ---
        # If the file has been modified very recently (e.g., within the last 15 seconds)
        if time.time() - f.stat().st_mtime < 15:
            size_pre = f.stat().st_size
            time.sleep(1)  # Brief wait to confirm the write operation
            size_post = f.stat().st_size
                
            if size_pre != size_post or size_post == 0:
                # If the size changes or is zero, the NVR is still working on it
                continue 
        # -------------------------------------
        cam_id_raw, ts = parse_filename(f, template, ts_format)
            
        if cam_id_raw is None or ts is None:
            continue
            
        # Normalization: Anything that comes out of parse_filename (e.g., "0", "1", "01")
        # we transform it into the "00", "01" format to match your cameras.json
        try:
            cam_id = str(cam_id_raw).zfill(2) 
        except:
            cam_id = cam_id_raw

        if cam_id not in index:
            index[cam_id] = []    
        #file_type = "video" if f.suffix.lower() == ".mp4" else "image"
        video_extensions = {".mp4", ".mkv", ".avi", ".mov"}
        file_type = "video" if f.suffix.lower() in video_extensions else "image"
      
        index[cam_id].append({
            "type": file_type,
            "timestamp": ts,
            "path": f,
            "cam_id": cam_id
        })

    # Sorting and validation
    for cam_id in index:
        # Sort by timestamp
        index[cam_id].sort(key=lambda x: x["timestamp"])
           
        # Quick stats
        vids = sum(1 for x in index[cam_id] if x["type"] == "video")
        imgs = sum(1 for x in index[cam_id] if x["type"] == "image")
        log.debug(f"Camera={cam_id}: video={vids} , img={imgs}.")
            
        if imgs < vids:
            log.warning(f"Camera={cam_id} has less images than videos")

    total_videos = sum(
        1 for cam_files in index.values() 
        for f in cam_files if f["type"] == "video"
    )
    total_images = sum(
        1 for cam_files in index.values() 
        for f in cam_files if f["type"] == "image"
    )
    log.info(f"Index complete: cameras={len(index)} | total_video={total_videos} | total_img={total_images}")
    return index
                  
def associate_files(index,input_dir):
    """Associate video files with NVR images from the provided file index.

    This function processes the existing index and creates associations between
    video timeline entries and corresponding NVR image captures for each camera.
    It is used to enrich raw results with additional metadata and frame references.

    Args:
        index (dict): Dictionary mapping camera_id to lists of file metadata entries.
                      Expected keys include 'type' (video/image) per file.
        input_dir (str | Path): Directory path where the association processing occurs.

    Returns:
        dict: A dictionary containing associations between video and NVR images.

    Side Effects:
        Logs progress messages during the association process.

    Note:
        This function assumes that `index` contains properly structured metadata 
        with 'type' field identifying whether an entry is a video or image file.
    """
    log.info(f"Starting associations videos-nvr images in folder={input_dir}")
    associations = {}

    for cam_id, timeline in index.items():
        # Separate by type (already sorted by timestamp from _build_index)
        videos = [x for x in timeline if x["type"] == "video"]
        images = [x for x in timeline if x["type"] == "image"]
        
        associations[cam_id] = []

        for i, video in enumerate(videos):
            video_ts = video["timestamp"]
            
            # Calculate boundary
            MAX_DELTA_SECONDS=180
            upper_bound = video_ts + timedelta(seconds=MAX_DELTA_SECONDS)
            
            # If there's a next video, restrict the boundary
            if i + 1 < len(videos):
                next_video_ts = videos[i+1]["timestamp"]
                if next_video_ts < upper_bound:
                    upper_bound = next_video_ts

            # Search for candidates (unassigned images in range)
            candidates = [
                img for img in images
                if not img.get("assigned", False)
                and video_ts <= img["timestamp"] < upper_bound
            ]

            # Create association record
            assoc_record = {
                "video_path": video["path"],
                "video_ts": video_ts,
                "nvr_images": [],
                "cam_id": cam_id
            }

            if candidates:
                # Marciamo le immagini come usate
                for img in candidates:
                    img["assigned"] = True
                
                assoc_record["nvr_images"] = [c["path"] for c in candidates]
                

            associations[cam_id].append(assoc_record)
    index_path = input_dir / "index.json"
    try:
    
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(associations, f, indent=4, ensure_ascii=False,default=str)
        
    except Exception as e:
        log.critical(f"Error during save on path={index_path}. err={e}")
        
    return associations

def parse_filename(path, template, ts_format):
    """
    Extracts camera_id (string) and timestamp (datetime) from Reolink files.
    Format: "NVR Name_ID_YYYYMMDDHHMMSS.mp4"
    """
    stem = path.stem
    
    # We transform the template into Regex (escape special characters like _)
    # We use re.escape to handle any dots or hyphens in the user template
    pattern = re.escape(template)
    pattern = pattern.replace(r"\{nvr_name\}", "(?P<nvr_name>.*?)")
    pattern = pattern.replace(r"\{camera_id\}", "(?P<camera_id>.*?)")
    pattern = pattern.replace(r"\{timestamp\}", "(?P<timestamp>\\d+)")
    
    match = re.search(f"^{pattern}$", stem)
    if not match:
        return None, None
        
    try:
        data = match.groupdict()
        cam_id = data.get("camera_id")
        timestamp_str = data.get("timestamp")
        
        timestamp = datetime.strptime(timestamp_str, ts_format)
        return cam_id, timestamp
    except (ValueError, IndexError, TypeError):
        return None, None
    
def sortVideos(settings,
               input_dir,
               output_dir,
               is_test,
               final_data,
               results,
               full_index):
    """Initialize FileSorter instance and launch sorting operation.

    Creates a FileSorter object with provided settings and directory paths,
    then executes the `sort_all` method with results data and full index.

    Args:
        settings (dict): Configuration dictionary for storage and processing.
        input_dir (str | Path): Source directory containing files to process.
        output_dir (str | Path): Destination directory for sorted results.
        is_test (bool): If True, use COPY mode; else use MOVE mode. Defaults to False.

    Returns:
        None: Executes sorting operation and returns after completion.

    Side Effects:
        Creates destination directories if needed.
        Moves/copies files based on configured method.
        Logs progress and any errors encountered during processing.
    """            
    # Init and start sorter
    file_sorter = FileSorter(settings,
                             input_dir, 
                             output_dir, 
                             is_test)
                
    file_sorter.sort_all(final_data, 
                         results, 
                         full_index)