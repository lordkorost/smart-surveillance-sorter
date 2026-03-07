# 🤖 YOLO Model Comparison

**Dataset:** February 17th — 102 videos + 90 images, Camera 00 + Camera 07 (garden/orchard, two overlapping angles)  
**Hardware:** AMD Ryzen 5 9600X | RX 9060 XT 16GB | ROCm 6.4 (Linux)  
**Pipeline:** `--mode full --refine --blip` — same config for all models  
**YOLO min conf:** Person 0.49, Vehicle 0.55, Animal 0.3  
**Note:** 54 videos already resolved by NVR images (YOLO img step) — only 48 videos processed in YOLO vid step for yolov8l. Other models resolve fewer images → more videos to process.

Each model was tested twice:
- **Default** — `FAKE_WEIGHTS` at default (all 1.0)
- **FK=0** — `FAKE_WEIGHTS` all set to 0.0 (no fake penalty, raw BLIP/CLIP scores)

---

## ⏱️ Performance

| Model | YOLO img | YOLO vid | BLIP | Total | Videos in YOLO vid |
|-------|----------|----------|------|-------|--------------------|
| yolov8n | 67.12 img/s — 00:01 | 3.56 s/vid — 04:16 | 02:00 | ~6 min | 72 (+24) |
| yolov8s | 32.15 img/s — 00:02 | 3.47 s/vid — 03:31 | 01:52 | ~6 min | 61 (+13) |
| yolov8m | 26.69 img/s — 00:03 | 3.55 s/vid — 03:15 | 01:41 | ~5 min | 55 (+7) |
| **yolov8l** ✅ | **16.28 img/s — 00:05** | **4.10 s/vid — 03:16** | **01:18** | **~5 min** | **48** |
| yolov8x | 9.35 img/s — 00:08 | 5.60 s/vid — 04:34 | 01:26 | ~6 min | 49 (+1) |

> ℹ️ **The YOLO img step matters.** Smaller models are faster per image but detect fewer persons → more videos pass to the YOLO vid step → overall pipeline is slower despite faster per-frame speed. yolov8n processes images 4x faster than yolov8l but ends up with 24 more videos to scan.

---

## 🎯 Accuracy — Default FAKE_WEIGHTS (all 1.0)

| Model | Person TP/FP/FN | Precision | Recall | Animal TP/FP/FN | Global Acc |
|-------|-----------------|-----------|--------|-----------------|------------|
| yolov8n | 79/2/1 | 97.5% | 98.8% | 0/0/7 | 92.16% |
| yolov8s | 79/3/1 | 96.3% | 98.8% | 2/0/5 | 94.12% |
| yolov8m | 80/3/0 | 96.4% | **100.0%** | 2/1/5 | 94.12% |
| **yolov8l** ✅ | **80/2/0** | **97.6%** | **100.0%** | **2/0/5** | **95.10%** |
| yolov8x | 80/3/0 | 96.4% | **100.0%** | 2/0/5 | 95.10% |

---

## 🎯 Accuracy — FAKE_WEIGHTS = 0.0 (no fake penalty)

| Model | Person TP/FP/FN | Precision | Recall | Animal TP/FP/FN | Global Acc |
|-------|-----------------|-----------|--------|-----------------|------------|
| yolov8n | 79/2/1 | 97.5% | 98.8% | 0/0/7 | 92.16% |
| yolov8s | 79/3/1 | 96.3% | 98.8% | 2/0/5 | 94.12% |
| yolov8m | 80/3/0 | 96.4% | **100.0%** | 3/2/4 | 94.12% |
| **yolov8l** ✅ | **80/2/0** | **97.6%** | **100.0%** | **3/0/4** | **96.08%** |
| yolov8x | 80/3/0 | 96.4% | **100.0%** | 2/0/5 | 95.10% |

---

## 📊 Key Findings

**yolov8l is the optimal model** — best balance of speed, person recall and false positives in both configurations.

**BLIP/CLIP is robust even without fake penalty.** Removing fake weights entirely (FK=0) causes minimal degradation across all models — only yolov8m introduces 2 animal FP. This demonstrates that the BLIP/CLIP scoring system is solid on its own: the underlying CLIP scores and BLIP captions already provide good discrimination even without additional tuning.

> 💡 On this specific camera, **yolov8l + FK=0 actually improves animal recall** (3 TP vs 2 TP) without introducing any false positives — the fake penalty was slightly over-penalizing a real animal. This is a good reminder that fake weights should be tuned per camera, not left at a global default.

**Why don't false positives explode without fake weights?**  
The crops saved by YOLO on this camera (trees, leaves, table, garden objects) are sufficiently different from person/animal prompts that CLIP scores remain low even without penalty. On cameras with more ambiguous backgrounds (ground, shadows, reflections) the fake penalty would matter much more — as shown in the [Camera 06 edge case](../edge-cases.md#-case-study-wood-piece-misclassified-as-bird--camera-06-garden-gate).

**yolov8n finds no animals at all** regardless of fake weights — the model simply misses small animals at this confidence threshold. Lowering animal min conf would help but would also increase false positives.

---

## ⚠️ Important Notes

1. **Confidence thresholds are model-dependent.** The same min conf values (P=0.49, V=0.55, A=0.3) were used for all models — optimal values differ per model. Switching models requires re-tuning confidence thresholds and BLIP fake weights for best results.

2. **These results are specific to this camera and scene.** A camera with more vehicles, more animals, or different lighting conditions may produce different relative rankings.

3. **yolov8x and tree trunks** — on cameras with trees, yolov8x may detect curved trunks as persons. The crops look convincingly human-shaped at 4K resolution. Fake weights mitigate this but higher confidence thresholds may be needed.

---

## 💡 Recommendation

**Use yolov8l (default)** unless you have a specific reason to switch:

- **CPU-only or low-VRAM systems** → try `yolov8n` or `yolov8s` for speed, accept slightly lower recall
- **Maximum person recall at any cost** → `yolov8x` with higher min conf to control false positives
- **Switching model** → always re-tune min conf and fake weights for that specific model on your cameras