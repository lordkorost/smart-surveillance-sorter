import logging
import os
from pathlib import Path
import shutil

from smart_surveillance_sorter.constants import FRAME_DIR
from smart_surveillance_sorter.utils import get_safe_path
log = logging.getLogger(__name__) 

class FileSorter:
    def __init__(self, settings, input_dir, work_dir, is_test=False):
        """
        Initialize the FileSorter object with configuration and directory paths.
        
        Configures whether to COPY or MOVE files based on test mode,
        sets up folder structure type (e.g., 'camera_first'), and determines
        if operations should be performed in-place.

        Args:
            settings (dict): Configuration dictionary containing storage settings.
            input_dir (str | Path): Input directory path where source data resides.
            work_dir (str | Path): Working/output directory for sorted files.
            is_test (bool): If True, use COPY mode; else use MOVE mode. Defaults to False.

        Attributes:
            settings (dict): Storage configuration dictionary.
            input_dir (Path): Input directory path.
            work_dir (Path): Work/output directory path.
            is_test (bool): Flag indicating test mode.
            inplace (bool): True if input_dir is within work_dir's hierarchy.
            method (str): "COPY" or "MOVE".
            structure_type (str): Folder structure type ('camera_first', etc.).
        """
        self.settings = settings
        self.input_dir = Path(input_dir)
        self.work_dir = Path(work_dir)
        self.is_test = is_test
        
        # Copy or move
        self.inplace = self.input_dir in self.work_dir.parents or self.input_dir == self.work_dir
        self.method = "COPY" if self.is_test else "MOVE"
        # Folder structure from settings
        storage_cfg = settings.get("storage_settings", {})
        self.structure_type = storage_cfg.get("structure_type", "camera_first")


        
    def _execute_io(self, src, dst):
        """
        Executes physical move or copy operation for files/folders.
        
        Handles file existence checks, creates output directories, and performs
        MOVE or COPY operations based on the configured method. Returns status
        codes to control continuation of processing cycle.

        Args:
            src (str | Path): Source file/directory path.
            dst (str | Path): Destination file/directory path.

        Returns:
            bool: True if operation succeeded, False otherwise.

        Raises:
            Exception: Errors are logged but not raised to avoid interrupting
                       the processing cycle of other items.
        """

        src = Path(src)
        dst = Path(dst)
        
        # 1. If input doesent exists  
        # Case Resume with MOVE, file is not anymore in input folder. 
        # Return false and continue with other videos.
        if not src.exists():
            return False
            
        # 2. If output dir exists
        # Case Resume with COPY
        if dst.exists():
            return True

        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if self.method == "MOVE":
                shutil.move(str(src), str(dst))
            else:
                shutil.copy2(str(src), str(dst))
            return True
        except Exception as e:
            # log.error to not interrupt cycle on others video
            log.error(f"Error on {self.method} of {src.name}.  error={e}")
            return False

    def _process_item(self, item, base_dir):
        """
        Process a single item record from JSON output or results list.
        
        Creates folder structure based on camera and category settings,
        moves/copies the main video file, and handles associated frames
        (including crop files if applicable).

        Args:
            item (dict): Record containing video_path, camera_name, category,
                         frames list, etc.
            base_dir (Path | str): Input directory or fallback path for processing.

        Returns:
            None: Operates in-place with side effects (file moves/copies).
        """
        video_src = Path(item["video_path"])
        
        # 1. Output folder
        target_dir = get_safe_path(
            base_dir, 
            item["camera_name"], 
            item["category"], 
            self.structure
        )
        target_dir.mkdir(parents=True, exist_ok=True)

        # 2. Move/Copy video
        dest_video = target_dir / video_src.name
        self._execute_io(video_src, dest_video)

        # 3. FRAME 
        # If frame for video exists, move
        if "frames" in item and item["frames"]:
            # Create subfolder for frame frame
            frames_target_dir = target_dir / "frames" / video_src.stem
            frames_target_dir.mkdir(parents=True, exist_ok=True)
            
            for frame_info in item["frames"]:
                frame_src = Path(frame_info["frame_path"])
                if frame_src.exists():
                    dest_frame = frames_target_dir / frame_src.name
                    self._execute_io(frame_src, dest_frame)
                    
                # If crop exists move
                if "crop_path" in frame_info and frame_info["crop_path"]:
                    crop_src = Path(frame_info["crop_path"])
                    if crop_src.exists():
                        dest_crop = frames_target_dir / crop_src.name
                        self._execute_io(crop_src, dest_crop)

   
    def sort_all(self, final_results, raw_results, full_index):
        """Process and organize classification results into the destination directory structure.
        
        This method organizes identified video files, associated frames, crop images, 
        and NVR images from a camera mapping. It handles missing video entries by marking them 
        as 'others', moves files to their correct target paths based on camera/category logic,
        and performs cleanup (removing empty directories/index files) if not in test mode.

        Args:
            final_results (list | dict): The list of items containing classification results,
                                         including video_path, camera_name, category, etc.
            raw_results (list | dict): Detailed raw results containing frame and crop paths
                                        linked to each video path.
            full_index (dict): A mapping of camera_id to entries, used to retrieve missing 
                               videos or NVR images associated with the current processing batch.

        Returns:
            None: Operates in-place, moving files and modifying directory structures.
            
        Side Effects:
            Creates destination directories if they don't exist (unless test mode).
            Removes empty frame/crop directories after file processing to save space.
            Handles cases where a camera has no corresponding video by marking it as 'others'.
        """
        if self.inplace:
            self.dest_base = self.input_dir
        else:
            self.dest_base = self.work_dir
            self.dest_base.mkdir(parents=True, exist_ok=True)
            if not os.access(self.work_dir, os.W_OK):
                self.dest_base = self.work_dir / "SMI_SORTED_RESULTS"
                self.dest_base.mkdir(parents=True, exist_ok=True)
                log.warning(f"Output dir not writeable! Results in folder={self.dest_base}")
        # import os
        # if os.access(self.input_dir, os.W_OK):
        #     # Caso normale: scriviamo nell'input
        #     self.dest_base = self.input_dir
        #     log.info(f"Dir={self.input_dir}")
        # else:
        #     self.dest_base = self.work_dir / "SMI_SORTED_RESULTS"
        #     self.dest_base.mkdir(parents=True, exist_ok=True)
        #     log.warning(f"Input is not writeable! Risults on folder={self.dest_base}")
        # --- loading real mapping id -> name ---
        from smart_surveillance_sorter.utils import get_camera_mapping 
        camera_mapping = get_camera_mapping() 
        


        classificati_paths = {item["video_path"] for item in final_results}

        # 2. Add missing videos from full_index
        for cam_id, entries in full_index.items():
            cam_id_str = str(cam_id).zfill(2) if str(cam_id).isdigit() else str(cam_id)
            cam_name = camera_mapping.get(cam_id_str, f"camera_{cam_id_str}")
            
            for entry in entries:
                v_path = str(entry["video_path"])
                if v_path not in classificati_paths:
                    # add as 'others'
                    final_results.append({
                        "camera_id": cam_id_str,
                        "camera_name": cam_name,
                        "video_name": Path(v_path).name,
                        "video_path": v_path,
                        "category": "others",
                        "confidence": 0,
                        "best_frame_path": None,
                        "engine": "none"
                    })
                    classificati_paths.add(v_path)

        files_processati = set() 

        for item in final_results:
            v_path = item["video_path"]
            cam_name = item["camera_name"]
            #cat = item["category"]
            
            # --- Category ---
            # IF IA return  'nothing', force in 'others'
            cat = item["category"]
            if cat == "nothing":
                cat = "others"
            target_dir = get_safe_path(self.dest_base, cam_name, cat, self.structure_type)
            
            # Move frames dir and crops
            if v_path not in files_processati:
                if self._execute_io(v_path, target_dir / item["video_name"]):
                    files_processati.add(v_path)

        
            video_details = next((r for r in raw_results if r["video_path"] == v_path), None)
            if video_details and "frames" in video_details:
                for f_info in video_details["frames"]:
                    
                    # Frame 
                    f_src = f_info.get("frame_path")
                    if f_src and f_src not in files_processati:
                        if self._execute_io(f_src, target_dir / Path(f_src).name):
                            files_processati.add(f_src)
                    
                    # Crop 
                    c_src = f_info.get("crop_path")
                    if c_src and c_src not in files_processati:
                        if self._execute_io(c_src, target_dir / Path(c_src).name):
                            files_processati.add(c_src)        

            # Move NVR img searching in full_index)
            # Find record in full_index for this video
            for cam_id, entries in full_index.items():
                entry = next((e for e in entries if str(e["video_path"]) == v_path), None)
                if entry and "nvr_images" in entry:
                    for img_p in entry["nvr_images"]:
                        img_src = str(img_p)
                        if img_src not in files_processati:
                            if self._execute_io(img_src, target_dir / Path(img_src).name):
                                files_processati.add(img_src)

                 
            # --- Remove frames dir empty and index.json---
            if not self.is_test:
                frames_dir = self.work_dir / FRAME_DIR
                if frames_dir.exists() and not any(frames_dir.iterdir()):
                    frames_dir.rmdir()
                    log.debug(f" Removed empty frames dir: {frames_dir}")
            index_file = self.input_dir / "index.json"
            if index_file.exists():
                index_file.unlink()
                log.debug(f" Removed index.json")
        

    def cleanup(self):
        """Remove the entire cache directory if operating in MOVE mode.

        This method deletes the work_dir folder if it exists and the active method 
        is 'MOVE'. It includes a safety check to prevent accidental deletion of the 
        user's input root directory or protected folders.

        Args:
            self: Instance of FileSorter class.

        Returns:
            None: Performs file system operations in-place.

        Side Effects:
            Removes all files from `self.work_dir`.
            Logs errors if removal fails or safety protection triggers.

        Note:
            Imports shutil at module level usually, but placed here for context 
            if not available globally. Ideally, move 'import shutil' to the top of the file.
        """
        if self.method == "MOVE" and self.work_dir.exists():
            # Safety check: Ensure we are only deleting work_dir and not input_dir itself
            if self.work_dir.resolve() == self.input_dir.resolve():
                log.error("Dir protection active. output_dir is input_dir. Cleanup abort.")
                return

            import shutil
            try:
                shutil.rmtree(self.work_dir)
                log.info(f"Cache folder removed. Folder={self.work_dir.name}")
            except Exception as e:
                log.error(f"Error removing cache: error={e}")