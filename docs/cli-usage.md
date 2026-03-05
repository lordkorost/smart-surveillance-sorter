## 🖥️ CLI Usage

After installation, the `sss` command is available in the virtual environment.

### Basic Usage
```bash
# Activate venv first
source .venv/bin/activate        # Linux
.venv\Scripts\activate           # Windows

# Only yolo no sorting only yolo res json (to continue later)
sss --dir /path/to/videos 

# Sort with BlipClip after Yolo 
sss --dir /path/to/videos --refine --blip

# Add fallback step after refine (use ollama and vision model)
sss --dir /path/to/videos --refine --blip --fallback

# Sort with Vision after Yolo
sss --dir /path/to/videos --refine --vision

# Sort in a different folder
sss --dir /path/to/videos -o /path/to/output --refine --blip

# Person mode only
sss --dir /path/to/videos --mode person

# Force to use cpu only yolo
sss --dir /path/to/videos --device cpu
```

### All Options

| Option | Short | Description |
|--------|-------|-------------|
| `--dir` | `-d` | Input directory (required) |
| `--mode` | `-m` | `full` / `person` / `person_animal` (default: `full`) |
| `--output-dir` | `-o` | Output directory (default: same as input) |
| `--vision` | | Use Ollama Vision instead of BLIP |
| `--blip` | | Use CLIP+BLIP engine (default) |
| `--refine` | | Enable refinement step |
| `--fallback` | | Enable Vision fallback for uncertain cases |
| `--check-clean` | | Enable during refine with vision to check cameras lens clean |
| `--device` | | `cuda` / `cpu` (default: auto-detect) |
| `--test` | | Test mode: copy instead of move, save metrics, debug log|
| `--no-sort` | | Analyze only, do not move files |
| `--ground` | | Generate ground truth JSON |
| `--compare` | | Compare results with ground truth |
| `--gt` | |  Path to ground truth JSON for compare |
| `--res` | | Path to results JSON for compare |

### Examples
```bash
# Full pipeline with Vision, test mode (no file moving)
sss --dir /mnt/nvr/2026-03-01 --mode full --refine --vision --test --no-sort

# Full pipeline with BlipClip, test mode (file copy)
sss --dir /mnt/nvr/2026-03-01 --mode full --refine --blip --test 

# Person only, GPU
sss --dir /mnt/nvr/cameras --mode person --refine --device cuda

# Person - animal, CPU, blip and fallback 
sss --dir /mnt/nvr/day-dir --mode person_animal --device cpu --refine --blip --fallback

# Yolo - Vision - Checkclean - Sorting (copy) in different output dir mode person using gpu
sss -d /path/to/video/dir -o /path/to/sorted/dir -m person -device cuda --refine --vision --check-clean --test
```

### Tools for test
```bash
# Generate ground truth for a folder sorted manually
sss --dir /mnt/nvr/2026-03-01 --ground

# Compare results with ground truth (ground_truth.json and classification_results.json in same folder)
sss --dir /mnt/nvr/2026-03-01 --compare

# Compare results with ground truth with custom name or in different folder and classification_results.json in folder
sss --dir /mnt/nvr/2026-03-01 --compare --gt /path/to/custom_ground.json

# Compare results with ground truth and classification_results with custom name or in different folder
sss --compare --gt /path/to/custom_ground.json --res /path/to/custom_results.json
```


### Real time sorter
Real time sorter cycle scan on a folder and sort files

| Option | Description |
|--------|-------------|
| `--dir` | Input directory (required) |
| `--mode` | `full` / `person` / `person_animal` (default: `full`) |
| `--output-dir`   | Output directory (default: same as input) |
| `--vision` |  Use Ollama Vision instead of BLIP |
| `--blip` |  Use CLIP+BLIP engine (default) |
| `--refine` |  Enable refinement step |
| `--device` |  `cuda` / `cpu` (default: auto-detect) |
| `--interval` |  (default: 60 secs) |

### Examples
```bash
# Scan every 60 secs for new videos with yolo - blip and sort, force gpu
sss-rt --dir /mnt/nvr/2026-03-01 --mode full --refine --blip --device cuda

# Scan every 180 secs for new videos with yolo - vision and sort in output dir, mode person
sss-rt --dir /mnt/nvr/2026-03-01 --output-dir /my/sort/dir --mode person --refine --vision --interval 180

# Scan every 90 secs for new videos with yolo - blip and sort, force cpu
sss-rt --dir /mnt/nvr/2026-03-01 --refine --vision --interval 90 --device cpu
```

### WebUI
```bash
sss-webui
```


