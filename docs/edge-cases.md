# 🔍 Edge Cases & Camera Tuning

This page documents real-world edge cases encountered during testing, with step-by-step diagnosis and configuration fixes. These cases illustrate the tuning workflow described in the [Testing Guide](testing-guide.md).

## Cases
- [🪵 Wood piece — fixing false positives with ignore_labels](#-case-study-wood-piece-misclassified-as-bird--camera-06-garden-gate)
- [🗿 Garden gnomes — Vision desc tuning](#-case-study-garden-gnomes-misclassified-as-persons--camera-04-balcony)
- [🐄 Livestock camera — selective animal ignoring](#-livestock-camera--selective-animal-ignoring)
- [🚗 Parking camera — vehicles as background](#-parking-camera--vehicles-as-background)

---

## 🪵 Case Study: Wood Piece Misclassified as Bird — Camera 06 (Garden Gate)

**Dataset:** February 13th — 84 videos + 70 images, Camera 06 (garden with gate, street visible)  
**Scene:** Wind-blown wood piece near the gate, scattered garden objects, shadows, direct sunlight  
**Hardware:** AMD Ryzen 5 9600X | RX 9060 XT 16GB | ROCm 6.4 (Linux)  
**Cameras:** Reolink 4K (20fps day / 12fps night)

---

### Step 1 — Initial Run (Default Config)

```
YOLO img:  15.16 img/s  00:04  (66 images)
YOLO vid:  5.92 s/vid   08:17  (84 videos)
BLIP:      1.85 vid/s   02:35  (84 videos)
Total:     ~11 min
```

| Category | TP | FP | FN | Precision | Recall |
|----------|----|----|----|-----------|--------|
| PERSON | 2 | 0 | 1 | 100.0% | 66.7% |
| VEHICLE | 8 | 0 | 3 | 100.0% | 72.7% |
| ANIMAL | 1 | 4 | 3 | 20.0% | 25.0% |
| OTHERS | 63 | 0 | 0 | 100.0% | 100.0% |
| **Global** | | | | **88.10%** | Avg: 54.80% |

4 false positive animals, 1 missed person, 3 missed vehicles.

---

### Step 2 — Inspect Extracted Frames → Identify Root Cause

Opening the extracted frames folder, animal crops clearly show a piece of wood — not an animal.

Checking `yolo_scan_res.json` confirms the pattern:

```json
{
    "category": "animal",
    "yolo_label": "bird",
    "confidence": 0.527,
    "area_ratio": 0.0036
}
```

**YOLO consistently classifies the wood piece as `bird` with confidence 0.5+.**

The fake weights (GROUND, GARDEN, WOOD, SHOE all at 1.0) reduce false positive frames significantly but cannot eliminate them entirely — some bird frames still pass through BLIP.

> 💡 **Why are we losing vehicles?** The wood piece triggers ANIMAL early exit — YOLO stops analyzing the video after 3 bird detections and moves to the next one, missing vehicles that appear later in the same video.

---

### Step 3 — Fix 1: Add `bird` to ignore_labels

```json
"06": {
    "filters": {
        "ignore_labels": ["bird"]
    },
    "blip_rules": {
        "FAKE_WEIGHTS": {
            "GROUND": 1.0,
            "GARDEN": 1.0,
            "SHOE": 1.0,
            "WOOD": 1.0
        }
    }
}
```

```
YOLO vid:  7.53 s/vid   10:32  (slower — no more false early exits on bird)
BLIP:      1.35 vid/s   01:02  (faster — fewer false animal frames to process)
Total:     ~11 min
```

| Category | TP | FP | FN | Precision | Recall |
|----------|----|----|----|-----------|--------|
| PERSON | 2 | 0 | 1 | 100.0% | 66.7% |
| VEHICLE | 9 | 0 | 2 | 100.0% | 81.8% |
| ANIMAL | 3 | 0 | 1 | 100.0% | 75.0% |
| OTHERS | 66 | 0 | 0 | 100.0% | 100.0% |
| **Global** | | | | **95.24%** | Avg: 74.49% |

✅ All animal false positives eliminated  
✅ 1 vehicle recovered  
✅ 2 animals recovered  
❌ Person FN still present

---

### Step 4 — Diagnose Remaining Person FN

Checking `clip_blip_res.json` for the missed person video:

```json
"clip_crop":  { "PERSON": 0.238 },
"clip_frame": { "PERSON": 0.009 },
"blip_caption": "there is a motorcycle that is parked on the side of the road",
"fake_scores": { "GROUND": 0.582 },
"final_scores": { "PERSON": 0.0 }
```

YOLO detected a **person's head visible above the gate** (confidence 0.51, bbox area = 0.07% of frame). CLIP partially recognizes the person (crop score 0.238) but the **GROUND fake score of 0.582 with weight 1.0 completely cancels the person score**.

The camera overlooks a garden — the ground occupies most of the frame — so GROUND fake penalty is too aggressive for this specific camera.

---

### Step 5 — Fix 2: Lower GROUND fake weight to 0.3

```json
"FAKE_WEIGHTS": {
    "GROUND": 0.3,
    "GARDEN": 1.0,
    "SHOE": 1.0,
    "WOOD": 1.0
}
```

| Category | TP | FP | FN | Precision | Recall |
|----------|----|----|----|-----------|--------|
| PERSON | 3 | 1 | 0 | 75.0% | **100.0%** |
| VEHICLE | 9 | 0 | 2 | 100.0% | 81.8% |
| ANIMAL | 3 | 0 | 1 | 100.0% | 75.0% |
| **Global** | | | | **95.24%** | Avg: 85.61% |

✅ Person recovered (100% recall)  
❌ 1 new false positive — YOLO saved a palm tree shadow as person, now passing the lowered threshold

---

### Step 6 — Fix 3: Fine-tune GROUND weight to 0.5

```json
"FAKE_WEIGHTS": {
    "GROUND": 0.5,
    "GARDEN": 1.0,
    "SHOE": 1.0,
    "WOOD": 1.0
}
```

| Category | TP | FP | FN | Precision | Recall |
|----------|----|----|----|-----------|--------|
| PERSON | 3 | 0 | 0 | **100.0%** | **100.0%** |
| VEHICLE | 9 | 0 | 2 | 100.0% | 81.8% |
| ANIMAL | 3 | 0 | 1 | 100.0% | 75.0% |
| OTHERS | 66 | 0 | 0 | 100.0% | 100.0% |
| **Global** | | | | **96.43%** | Avg: 85.61% |

✅ Zero false positives on persons  
✅ Zero false negatives on persons  
The palm shadow video was correctly moved to OTHERS — vehicle frame scores were not high enough to override (acceptable: it's a distant vehicle outside the gate, not a security concern).

---

### Step 7 — Optional: Add Fallback Vision for Maximum Animal Recall

```bash
sss --dir /path --mode full --refine --blip --fallback --test --no-sort
```

```
Fallback Vision:  17.63 s/vid  03:13  (11 uncertain videos)
Total:            ~14 min
```

| Category | TP | FP | FN | Precision | Recall |
|----------|----|----|----|-----------|--------|
| PERSON | 3 | 0 | 0 | **100.0%** | **100.0%** |
| VEHICLE | 9 | 0 | 2 | 100.0% | 81.8% |
| ANIMAL | 4 | 0 | 0 | **100.0%** | **100.0%** |
| OTHERS | 66 | 0 | 0 | 100.0% | 100.0% |
| **Global** | | | | **97.62%** | Avg: 93.94% |

✅ Last missed animal found by Vision  
The 2 remaining vehicle FNs are not recoverable — YOLO saved too many false person frames on those videos and vehicle scores cannot override. Not considered a real issue: these are vehicles visible outside the gate (street), which should be masked on the NVR anyway.

> ℹ️ Note on remaining vehicle FNs: filming public streets may raise privacy/legal concerns depending on your jurisdiction. These videos were recorded only for testing purposes and deleted immediately after.

---

### 📊 Complete Progression

| Step | Config | Global Acc | Person recall | Animal recall | Vehicle recall |
|------|--------|------------|---------------|---------------|----------------|
| 1 | Default | 88.10% | 66.7% | 25.0% | 72.7% |
| 2 | + ignore bird | 95.24% | 66.7% | 75.0% ✅ | 81.8% ✅ |
| 3 | + GROUND=0.3 | 95.24% | 100.0% ✅ | 75.0% | 81.8% |
| 4 | + GROUND=0.5 | 96.43% | **100% / 100%** ✅ | 75.0% | 81.8% |
| 5 | + Fallback | **97.62%** | **100% / 100%** ✅ | **100%** ✅ | 81.8% |

---

### Final Configuration for Camera 06

```json
"06": {
    "name": "Giardino Cancello",
    "location": "outdoor",
    "search_patterns": ["_06_", "ch06", "Cam06"],
    "priority": "person",
    "desc": "Garden area, street visible through the gate.",
    "dynamic_stride": false,
    "filters": {
        "ignore_labels": ["bird"]
    },
    "thresholds": {
        "person": 0.50,
        "vehicle": 0.55,
        "animal": 0.3
    },
    "blip_rules": {
        "FAKE_WEIGHTS": {
            "GROUND": 0.5,
            "GARDEN": 1.0,
            "SHOE": 1.0,
            "WOOD": 1.0
        }
    }
}
```

---

## 🧹 The Real Fix

> **The most effective fix for environment-triggered false positives is removing the source.**  
> The wood piece was brought by wind — once removed, all bird false positives disappeared without any configuration change. `ignore_labels` is a software workaround for a physical problem.  
> Keep cameras clean and the monitored area tidy for best results. 😄

---

---

## 🗿 Case Study: Garden Gnomes Misclassified as Persons — Camera 04 (Balcony)

**Dataset:** February 13th — 97 videos + 81 images, Camera 04 (kitchen balcony overlooking garden)  
**Scene:** Garden visible from balcony, 3-4 garden gnomes on the lawn visible in frame

---

### The Problem — Vision Without `desc`

Without camera description, Vision sees small human-shaped figures in the garden and classifies them as persons. 4 false positives across the dataset.

```
YOLO img:  16.52 img/s  00:04  (81 images)
YOLO vid:  4.49 s/vid   05:59  (80 videos)
Vision:    1.13 s/vid   01:25  (97 videos)
Total:     ~6 min
```

| Category | TP | FP | FN | Precision | Recall |
|----------|----|----|----|-----------|--------|
| PERSON | 25 | 5 | 0 | 83.3% | 100.0% |
| OTHERS | 66 | 0 | 0 | 100.0% | 100.0% |
| **Global** | | | | **94.85%** | — |

Vision reasoning on a gnome video (no desc):
> *"In the garden area, there are **two small figures that look like people**. They are near the orange trees. So according to rule 1, if a person is present, output 'person'."*  
> 🎯 **FINAL VERDICT: person**

Same pattern across all 4 FP videos — Vision identifies "small figures that look like people" and immediately outputs person without questioning.

---

### The Fix — Add `desc` to cameras.json

```json
"04": {
    "desc": "Balcony area, with garden underneath. There are some garden gnomes on the lawn."
}
```

Vision reasoning on the same video (with desc):
> *"Looking at the image, there's a balcony with a garden below. In the garden, there are some small figures — wait, the description says **garden gnomes**, but are they people? Wait, **garden gnomes are not real people**. Wait, but maybe there's a person. Let me check again [...] The garden has **garden gnomes (statues)**, so no real people. No animals, no vehicles. So the output should be 'nothing'."*  
> 🎯 **FINAL VERDICT: others**

The desc triggers a visible internal conflict in the reasoning — the model "fights" between rule 1 ("output person immediately") and the camera context ("garden gnomes are statues"). The context wins every time.

All 4 reasoning examples, before and after:

| Video | Without desc | With desc |
|-------|-------------|-----------|
| 125054 | *"two small figures that look like people"* → **person** | *"garden gnomes (statues), so no real people"* → **others** |
| 124702 | *"two small figures that look like people"* → **person** | *"gnomes are not people [...] output nothing"* → **others** |
| 125146 | *"I can see a couple of people"* → **person** | *"garden gnomes. Wait, gnomes are not people"* → **others** |
| 143459 | *"there's a person in the garden"* → **person** | *"garden gnomes, which are statues, not people"* → **others** |

---

### Results With `desc`

```
Vision:  1.08 s/vid  01:44  (97 videos)
Total:   ~7 min
```

| Category | TP | FP | FN | Precision | Recall |
|----------|----|----|----|-----------|--------|
| PERSON | 25 | 1 | 0 | 96.2% | **100.0%** |
| OTHERS | 71 | 0 | 0 | 100.0% | 100.0% |
| **Global** | | | | **98.97%** | — |

4 false positives resolved. The 1 remaining FP has YOLO confidence = 1.0 — the safety clause overrides Vision's "nothing" verdict and keeps it as PERSON. Acceptable behavior: when YOLO is that confident, we trust it.

> 💡 **`desc` is the most powerful Vision tuning parameter.** A single sentence describing permanent scene elements (statues, decorations, fixed objects) can eliminate entire categories of false positives without touching thresholds or weights.

---

## 🐄 Livestock Camera — Selective Animal Ignoring

On a camera monitoring a stable or pen, animals are always present — they are background, just like parked cars in a parking area. The goal is to detect intruders or predators, not the animals that are supposed to be there.

Using `ignore_labels` selectively allows ignoring *known* animals while keeping detection active for *unexpected* ones:

```json
"stable_cam": {
    "filters": {
        "ignore_labels": ["cow", "sheep", "horse"]
    }
}
```

**Result:** cows, sheep and horses are ignored. A person entering the stable, a bear, or a stray dog will still trigger an alert. Even a fox — which YOLO may classify as `dog` or `cat` — will still be detected as ANIMAL.

> ℹ️ Any real threat to livestock requires a physical presence that SSS will catch — either as PERSON or as an unexpected ANIMAL.

---

## 🚗 Parking Camera — Vehicles as Background

A parking camera has vehicles in frame at all times — they are background, just like the lawn in a garden or the animals in a stable. The goal is not to detect parked cars but to detect anyone entering or leaving the area.

SSS does not detect movement and cannot tell if a car is new or has moved. However, **any vehicle entering or leaving must be driven by a person** — who will be seen getting in or out. Detecting that person is equivalent to detecting the vehicle event, without the false positives and processing overhead that static parked cars generate.

```json
"parking_cam": {
    "filters": {
        "ignore_labels": ["car", "truck", "bus", "motorcycle", "bicycle", "boat", "train"],
        "ignore_classes": ["VEHICLE"]
    }
}
```

This has two important side effects:

**1. Dynamic stride activates correctly.** With vehicles always in frame, YOLO never triggers early exit on VEHICLE and dynamic stride never activates — YOLO stays at `stride=0.6s` for every video. With vehicle labels ignored, dynamic stride works normally and speeds up YOLO significantly on empty videos.

**2. BLIP processes far fewer frames.** In `--mode full`, YOLO saves up to 9 frames per video (3 per category). With vehicles ignored, only person and animal frames are saved — BLIP time drops dramatically on large parking datasets.

> 📖 See [Mode Comparison](benchmarks/mode-comparison.md) for benchmark results showing the full impact of this configuration on a 376-video parking dataset.