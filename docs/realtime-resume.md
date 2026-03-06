# ⏯️ Real-Time Mode & Resume

## 🧠 The Core Concept

Both real-time and resume share the same principle — **the system never re-processes what it has already seen**.

On every run, results are saved incrementally to JSON cache files inside the output directory. On the next run (or the next real-time cycle), the system checks these files and automatically skips already-processed videos.

This means:
- **Resume** — an interrupted scan continues exactly from where it stopped
- **Real-time** — each cycle only processes videos added since the last run
- **New videos in completed folder** — running again on a finished folder only processes the new additions

---

## ⏩ Resume

If a scan is interrupted (Ctrl+C, crash, power loss), simply re-run the same command:

```bash
sss --dir /path/to/footage --mode full --refine --blip --test
```

The system will:
1. Load existing `yolo_scan_res.json` → skip already-scanned videos
2. Load existing `clip_blip_res.json` → skip already-refined videos
3. Continue from the first unprocessed video

### Adding New Videos to a Completed Folder

If new videos are added to a folder that was already fully processed, just re-run the same command — only the new videos will be processed.

### ⚠️ Important Rules for Resume

> **Never change the engine between runs on the same folder.**  
> Mixing `--blip` and `--vision` results on the same folder causes incorrect classifications. The system will warn you if it detects a mismatch — do not bypass this warning.

> **Settings changes between runs are applied only to new videos.**  
> If the first 100 videos were processed with `vid_stride_sec=0.6` and you change it to `1.0` before resuming, the remaining videos will use the new setting. Results will be inconsistent. Change settings only before starting a fresh run.

> **To force a full re-scan**, delete the relevant cache file:
> | Delete this file | Effect |
> |-----------------|--------|
> | `json/yolo_scan_res.json` | Forces full YOLO re-scan |
> | `json/clip_blip_res.json` | Forces BLIP re-processing |
> | `json/vision_cache.json` | Forces Vision re-processing |
> | All files | Complete fresh start |

---

## 🔄 Real-Time Mode

Real-time mode runs continuous scan cycles at a defined interval, processing only new videos each time.

```bash
sss-rt --dir /path/to/footage --mode full --refine --blip --interval 60
```

### How It Works

1. Scan runs on the folder
2. New videos are sorted and moved
3. System waits `--interval` seconds
4. On the next cycle, only videos added since the last run are processed
5. Repeat until stopped with Ctrl+C

### New Video Detection

The system uses a **safe detection mechanism** to avoid processing incomplete files:
- Checks the file's last modified timestamp
- Verifies the file size is stable for several seconds before processing
- This prevents processing files that are still being written by the NVR

### Real-Time Mode Options

| Option | Default | Description |
|--------|---------|-------------|
| `--dir` | required | Input directory to watch |
| `--output-dir` | same as input | Output directory for sorted videos |
| `--mode` | `full` | Classification mode: `full` / `person` / `person_animal` |
| `--device` | `cuda` | `cuda` / `cpu` |
| `--interval` | `60` | Seconds to wait between scan cycles |
| `--refine` | required | Enable BLIP or Vision refinement |
| `--blip` / `--vision` | blip | Refinement engine |

> ⚠️ The following options are **not supported** in real-time mode:
> - `--test` — files must be moved, not copied
> - `--no-sort` — same reason
> - `--fallback` — not compatible with continuous processing

### Example: Daily NVR Monitoring

A typical home NVR setup records continuously into a daily folder via FTP/Samba. Real-time mode integrates naturally:

```bash
# Start at 8 AM — first cycle processes all videos from midnight to now
# Subsequent cycles process only new videos as they arrive
sss-rt --dir /mnt/nvr/2026-03-06 --mode full --refine --vision --interval 120
```

**Workflow:**
- Wake up at 8 AM, mount the NVR Samba share, launch `sss-rt`
- First run processes all videos from midnight to 8 AM
- Every 2 minutes, new videos are processed as they arrive from the NVR
- Stop anytime with Ctrl+C — resume is automatic next time
- At midnight, start a new session on the new day's folder

> ℹ️ You can switch between real-time and normal mode on the same folder. Running `sss-rt` after a normal `sss` run (or vice versa) works correctly — the resume mechanism is shared.

---

## ⚠️ What to Avoid

| Action | Risk | Why |
|--------|------|-----|
| Changing `--blip` to `--vision` mid-folder | ❌ Incorrect results | Cache files are engine-specific |
| Modifying settings between resume runs | ⚠️ Inconsistent results | New settings apply only to unprocessed videos |
| Deleting sorted videos before the run completes | ❌ Files may be re-sorted | The system tracks what was processed, not what exists |
| Running two instances on the same folder | ❌ Race condition | JSON files will be corrupted |
| Using `--test` with real-time | ❌ Not supported | Files are never moved → infinite re-processing loop |