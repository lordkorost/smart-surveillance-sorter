# Mode Comparison

**Dataset:** Folder March 5th — 376 videos + 298 images, single camera (Parking/Camera 02)  
**Scene:** Spider web on camera lens, 6+ hours of continuous recording, mostly empty 3-minute videos. 3 cars always parked → no OTHERS category.  
**Hardware:** AMD Ryzen 5 9600X | RX 9060 XT 16GB | ROCm 6.4 (Linux) | NVMe storage  
**Models:** YOLOv8l, BLIP large, CLIP ViT-L-14  
**YOLO min conf:** Person 0.49, Vehicle 0.55, Animal 0.3  
**Dynamic Stride:** enabled on parking camera

---

## Tested Configurations

| Config | Mode | ignore_labels | Notes |
|--------|------|--------------|-------|
| A | `full` | none | Baseline — search everything |
| B | `person` | none | Search persons only |
| C | `person_animal` | none | Search persons and animals |
| D | `full` | vehicle labels in cameras.json | Per-camera vehicle exclusion |

---

## Performance

| Config | YOLO img | YOLO vid | BLIP | Total |
|--------|----------|----------|------|-------|
| A — full | 00:16 | 32:39 (5.55s/vid) | 14:11 | ~47 min |
| B — person | 00:16 | 26:14 (4.46s/vid) ✅ | 00:59 ✅ | ~27 min |
| C — person_animal | 00:16 | 26:34 (4.52s/vid) ✅ | 01:12 | ~28 min |
| D — full + ignore | 00:16 | 25:40 (4.36s/vid) ✅ | 01:10 | ~27 min |

>[!NOTE]
> **Why is `--mode full` significantly slower?**  
> Vehicle detections prevent dynamic stride from activating — YOLO stays at `stride=0.6s` instead of switching to `stride_fast=1.0s`. With vehicle labels ignored (Config D), dynamic stride activates normally, matching the speed of `--mode person`.

>[!NOTE]
> **Why is BLIP so much slower in Config A (14 min vs ~1 min)?**  
> In `--mode full`, YOLO saves up to 3 frames per category × 3 categories = 9 frames + 9 crops per video in the worst case. With `--mode person`, only person frames are saved → ~1 frame per video on average.

---

## Accuracy

### Config A — `--mode full`, no ignore_labels

| Category | TP | FP | FN | Precision | Recall |
|----------|----|----|----|-----------|--------|
| PERSON | 54 | 2 | 0 | 96.4% | **100.0%** |
| VEHICLE | 305 | 0 | 9 | 99.3% | 97.1% |
| ANIMAL | 5 | 0 | 3 | 100.0% | 62.5% |
| OTHERS | 0 | 0 | 0 | — | — |
| **Global** | | | | **94.99%** | Avg: 86.5% |

>[!NOTE]
> No OTHERS category — 3 cars always present in frame means every video has at least one vehicle detection.

---

### Config B — `--mode person`

| Category | TP | FP | FN | Precision | Recall |
|----------|----|----|----|-----------|--------|
| PERSON | 53 | 1 | 1* | 98.1% | 98.1% |
| VEHICLE | — | — | — | not searched | — |
| ANIMAL | — | — | — | not searched | — |
| **Person only** | | | | **98.1% precision** | **98.1% recall** |

>[!NOTE]
> *FN: person on tractor passing outside the gate at ~0:35, visible for <1s. Dynamic stride already active (pre_roll=20s elapsed). Not considered a real security false negative — subject is outside the monitored area. In a properly configured system this area would be masked on the NVR.

---

### Config C — `--mode person_animal`

| Category | TP | FP | FN | Precision | Recall |
|----------|----|----|----|-----------|--------|
| PERSON | 53 | 2 | 1* | 96.4% | 98.1% |
| VEHICLE | — | — | — | not searched | — |
| ANIMAL | 5 | 0 | 3 | 100.0% | 62.5% |
| **Avg Precision** | | | | **98.2%** | |
| **Avg Recall** | | | | | **80.3%** |

>[!NOTE]
> *Same FN as Config B — tractor person.

---

### Config D — `--mode full` + `ignore_labels` vehicle in cameras.json ✅ Recommended

```json
"filters": {
    "ignore_labels": ["car", "truck", "bus", "motorcycle", "bicycle", "boat", "train"],
    "ignore_classes": ["VEHICLE"]
}
```

| Category | TP | FP | FN | Precision | Recall |
|----------|----|----|----|-----------|--------|
| PERSON | 53 | 1 | 1* | 98.1% | 98.1% |
| VEHICLE | — | — | — | ignored via cameras.json | — |
| ANIMAL | 5 | 0 | 3 | 100.0% | 62.5% |
| **Avg Precision** | | | | **99.05%** | |
| **Avg Recall** | | | | | **80.3%** |

>[!NOTE]
> *Same FN as Config B — tractor person.
> **Why ignore vehicles on a parking camera?**  
> In a parking area, vehicles are part of the background — like a house or a tree. We're not interested in the vehicles themselves, but in whether someone enters the parking area. SSS does not detect movement and cannot tell if a car is new or has moved. However, any vehicle entering or leaving must be driven by a person — who will be seen getting in or out. Detecting that person is equivalent to detecting the vehicle event, without the false positives and processing overhead that static parked cars generate.


**Another example — barn or livestock camera:**  
A camera monitoring a stable always has livestock in frame — they are background, just like parked cars. The goal is to detect intruders or predators, not the animals that are always there.  
Using `ignore_labels` selectively allows ignoring *known* animals while keeping detection active for *unexpected* ones:

```json
"filters": {
     "ignore_labels": ["cow", "sheep", "horse"]
 }
```

Result: cows, sheeps and horses are ignored. A person entering the stable, a bear, or a stray dog will still be detected. Even a fox — which YOLO may classify as `dog` or `cat` — will still trigger an ANIMAL alert.  
Any real threat requires a physical presence that SSS will catch.

---

## Summary

| Config | Time | Person Recall | Animal Recall | Vehicle | Notes |
|--------|------|---------------|---------------|---------|-------|
| A — full | ~47 min | 100.0% | 62.5% | ✅ 97.1% | Slow, finds everything |
| B — person | ~27 min | 98.1% | ❌ not searched | ❌ | Fastest, persons only |
| C — person_animal | ~28 min | 98.1% | 62.5% | ❌ | Fast, no vehicles |
| D — full + ignore | ~27 min ✅ | 98.1% | 62.5% | ❌ per-camera | **Best balance** |

---

## Key Takeaways

1. **`--mode full` without configuration is the slowest** — vehicle detections prevent dynamic stride and force BLIP to process 9 frames/video instead of ~3
2. **`--mode person` is fastest** — but misses animals entirely
3. **`--mode full` + per-camera `ignore_labels`** achieves the same result as `--mode person_animal` but with **better precision** (99.05% vs 98.2%) and **faster YOLO** — recommended approach for mixed-camera setups
4. **Dynamic stride interaction** — any camera where the target category is always present (e.g. parked cars) will never activate fast stride in `--mode full`. Use `ignore_labels` to let dynamic stride work correctly
5. **BLIP time scales with frames saved** — `--mode full` = up to 9 frames/video, `--mode person` = ~1 frame/video. For large datasets this difference is significant (14 min vs 1 min on this test)
6. **0 missed persons on Camera 02** in Config A (mode full) — the tractor FN in Config B/C/D is caused by dynamic stride activating, not by mode selection

>[!TIP]
>  **Recommended setup for a parking camera:** `--mode full` globally + `ignore_classes: ["VEHICLE"]` and `ignore_labels: [car, truck, ...]` in `cameras.json` for the parking camera. This gives you per-camera control without limiting other cameras that may need vehicle detection.
