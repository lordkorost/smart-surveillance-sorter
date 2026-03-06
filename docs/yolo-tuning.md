# ⚙️ YOLO Parameter Tuning

**Dataset:** Folder February 17th — 399 videos + 505 images  
**Scene:** Rain, lightning, 6+ hours of continuous recording. Camera "Parking" with ~50% empty 3-minute videos.  
**Hardware:** AMD Ryzen 5 9600X | RX 9060 XT 16GB | ROCm 6.4 (Linux) | Videos on Samba share  
**Models:** YOLOv8l, BLIP large, CLIP ViT-L-14, qwen3-vl:8b  
**YOLO min conf:** Person 0.49, Vehicle 0.55, Animal 0.3 (same for all cameras, day and night)

> ℹ️ Dynamic Stride enabled only on "Parking" camera (large area, mostly empty footage).  
> ℹ️ False positives on PERSON in this dataset are insects on the camera lens and rain reflections — not misclassified humans.

---

## 📋 Tested Configurations

| Config | stride | num_occ | time_gap | warmup | pre_roll | stride_fast |
|--------|--------|---------|----------|--------|----------|-------------|
| **Default** | 0.6s | 3 | 3s | 5s | 20s | 1.0s |
| **Test 2** | 1.0s | 3 | 3s | 3s | 15s | 1.0s |
| **Test 3** | 1.0s | 2 | 5s | 3s | 10s | 1.0s |
| **Test 4** | 1.2s | 2 | 5s | 2s | 10s | 1.5s |

---

## ⏱️ Performance

| Config | YOLO img | YOLO vid | BLIP | Total (BLIP) | Total (Vision) |
|--------|----------|----------|------|--------------|----------------|
| Default | 00:33 | 46:20 | 02:54 | ~50 min | ~60 min |
| Test 2 | 00:31 | 41:09 | 02:52 | ~45 min | ~55 min |
| Test 3 | 00:31 | 40:16 | 02:00 | ~43 min | ~55 min |
| Test 4 | 00:31 | 39:38 | 01:55 | ~42 min | ~50 min |

> ℹ️ BLIP time decreases with lower `num_occurrence` — fewer saved frames per video = less BLIP work.

---

## 🎯 Accuracy — YOLO + BLIP

| Config | Global Acc | Person FP | Animal Recall | Vehicle Recall |
|--------|------------|-----------|---------------|----------------|
| Default | 94.99% | 13 | 46.7% | 93.3% |
| Test 2 | 94.99% | 12 | 40.0% | 93.3% |
| Test 3 | 94.49% | 12 | 40.0% | 80.0% |
| Test 4 | 95.49% | 8 | 40.0% | 80.0% |

---

## 🎯 Accuracy — YOLO + BLIP + Fallback

| Config | Global Acc | Person FP | Animal Recall | Vehicle Recall |
|--------|------------|-----------|---------------|----------------|
| Default | 96.24% | 13 | 80.0% | 93.3% |
| Test 2 | 96.24% | 12 | 73.3% | 93.3% |
| Test 3 | 96.49% | 12 | 80.0% | 93.3% |
| Test 4 | 96.49% | 8 | 80.0% | 93.3% |

---

## 🎯 Accuracy — YOLO + Vision ✅ Recommended

| Config | Global Acc | Person FP | Animal Recall | Vehicle Recall |
|--------|------------|-----------|---------------|----------------|
| Default | 96.99% | 9 | 80.0% | 100% |
| Test 2 | **98.75%** | **3** ✅ | **86.7%** ✅ | 100% |
| Test 3 | **98.75%** | **3** ✅ | **86.7%** ✅ | 100% |
| Test 4 | 98.50% | 4 | **86.7%** ✅ | 100% |

---

## 📊 Detailed Results

### Default params

#### YOLO + BLIP
| Category | TP | FP | FN | Precision | Recall |
|----------|----|----|----|-----------|--------|
| PERSON | 80 | 13 | 0 | 86.0% | **100.0%** |
| VEHICLE | 14 | 0 | 1 | 100.0% | 93.3% |
| ANIMAL | 7 | 0 | 8 | 100.0% | 46.7% |
| OTHERS | 278 | 0 | 0 | 100.0% | 100.0% |
| **Global** | | | | **94.99%** | Avg: 80.00% |

#### YOLO + BLIP + Fallback
| Category | TP | FP | FN | Precision | Recall |
|----------|----|----|----|-----------|--------|
| PERSON | 80 | 13 | 0 | 86.0% | **100.0%** |
| VEHICLE | 14 | 0 | 1 | 100.0% | 93.3% |
| ANIMAL | 12 | 0 | 3 | 100.0% | 80.0% |
| OTHERS | 278 | 0 | 0 | 100.0% | 100.0% |
| **Global** | | | | **96.24%** | Avg: 91.11% |

#### YOLO + Vision
| Category | TP | FP | FN | Precision | Recall |
|----------|----|----|----|-----------|--------|
| PERSON | 80 | 9 | 0 | 89.9% | **100.0%** |
| VEHICLE | 15 | 0 | 0 | 100.0% | 100.0% |
| ANIMAL | 12 | 0 | 3 | 100.0% | 80.0% |
| OTHERS | 280 | 0 | 0 | 100.0% | 100.0% |
| **Global** | | | | **96.99%** | Avg: 93.33% |

---

### Test 2 — Faster Stride (stride=1.0, warmup=3, pre_roll=15)

#### YOLO + BLIP
| Category | TP | FP | FN | Precision | Recall |
|----------|----|----|----|-----------|--------|
| PERSON | 80 | 12 | 0 | 87.0% | **100.0%** |
| VEHICLE | 14 | 0 | 1 | 100.0% | 93.3% |
| ANIMAL | 6 | 0 | 9 | 100.0% | 40.0% |
| OTHERS | 279 | 0 | 0 | 100.0% | 100.0% |
| **Global** | | | | **94.99%** | Avg: 77.78% |

#### YOLO + Vision ✅
| Category | TP | FP | FN | Precision | Recall |
|----------|----|----|----|-----------|--------|
| PERSON | 80 | 3 | 0 | **96.4%** | **100.0%** |
| VEHICLE | 15 | 0 | 0 | 100.0% | 100.0% |
| ANIMAL | 13 | 0 | 2 | 100.0% | **86.7%** |
| OTHERS | 286 | 0 | 0 | 100.0% | 100.0% |
| **Global** | | | | **98.75%** 🏆 | Avg: 95.56% 🏆 |

---

### Test 3 — Faster + Less Occurrences (stride=1.0, num_occ=2, time_gap=5, warmup=3, pre_roll=10)

#### YOLO + BLIP
| Category | TP | FP | FN | Precision | Recall |
|----------|----|----|----|-----------|--------|
| PERSON | 80 | 12 | 0 | 87.0% | **100.0%** |
| VEHICLE | 12 | 0 | 3 | 100.0% | 80.0% |
| ANIMAL | 6 | 0 | 9 | 100.0% | 40.0% |
| OTHERS | 279 | 0 | 0 | 100.0% | 100.0% |
| **Global** | | | | **94.49%** | Avg: 73.33% |

#### YOLO + Vision ✅
| Category | TP | FP | FN | Precision | Recall |
|----------|----|----|----|-----------|--------|
| PERSON | 80 | 3 | 0 | **96.4%** | **100.0%** |
| VEHICLE | 15 | 0 | 0 | 100.0% | 100.0% |
| ANIMAL | 13 | 0 | 2 | 100.0% | **86.7%** |
| OTHERS | 286 | 0 | 0 | 100.0% | 100.0% |
| **Global** | | | | **98.75%** 🏆 | Avg: 95.56% 🏆 |

---

### Test 4 — Maximum Speed (stride=1.2, num_occ=2, time_gap=5, warmup=2, pre_roll=10, stride_fast=1.5)

#### YOLO + BLIP
| Category | TP | FP | FN | Precision | Recall |
|----------|----|----|----|-----------|--------|
| PERSON | 80 | 8 | 0 | 90.9% | **100.0%** |
| VEHICLE | 12 | 0 | 3 | 100.0% | 80.0% |
| ANIMAL | 6 | 0 | 9 | 100.0% | 40.0% |
| OTHERS | 283 | 0 | 0 | 100.0% | 100.0% |
| **Global** | | | | **95.49%** | Avg: 73.33% |

#### YOLO + Vision
| Category | TP | FP | FN | Precision | Recall |
|----------|----|----|----|-----------|--------|
| PERSON | 80 | 4 | 0 | 95.2% | **100.0%** |
| VEHICLE | 15 | 0 | 0 | 100.0% | 100.0% |
| ANIMAL | 13 | 0 | 2 | 100.0% | **86.7%** |
| OTHERS | 286 | 0 | 0 | 100.0% | 100.0% |
| **Global** | | | | **98.50%** | Avg: 95.56% |

---

## 💡 Key Takeaways

1. **Vision consistently outperforms BLIP** on this dataset — especially for reducing Person FP (insects, rain reflections) and improving Animal recall (large white dog)
2. **Test 2 + Vision is the optimal combination** — 5 min faster than default with significantly better accuracy (98.75% vs 96.99%)
3. **Test 3 achieves identical accuracy to Test 2** with 2 fewer min — `num_occurrence=2` helps BLIP but Vision compensates for any YOLO misses
4. **Test 4** saves another 5 min but slightly worse Person precision (4 FP vs 3) — acceptable tradeoff for time-sensitive use cases
5. **BLIP alone struggles on this dataset** — Animal recall 40-47% due to large white dog misclassified as person. Fallback or Vision required.
6. **0 missed persons (FN=0) across all configurations** — person detection is robust even with aggressive stride settings

> ⚠️ **Important caveat:** These results apply to this specific dataset (slow-moving subjects, mostly empty footage). For cameras with fast-moving subjects (running person, fast vehicles), `vid_stride_sec=0.6` default is recommended to avoid missed detections. See [Edge Cases](edge-cases.md).