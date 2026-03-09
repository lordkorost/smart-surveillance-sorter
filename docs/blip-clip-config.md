# CLIP+BLIP Engine Tuning Guide

Settings are stored in `config/clip_blip_settings.json`. Most values can be overridden per-camera in `config/cameras.json` under `blip_rules`.

>[!NOTE]
> These settings are for the `--blip` engine only. Vision mode (`--vision`) uses different parameters — see [Tuning Guide](tuning-guide.md).

---

## Models

```json
"clip_model": "ViT-L-14",
"clip_pretrained": "openai",
"blip_model": "Salesforce/blip-image-captioning-large"
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `clip_model` | `ViT-L-14` | CLIP vision encoder. `ViT-L-14` is recommended — `ViT-H-14` is larger but performs worse on surveillance footage. |
| `clip_pretrained` | `openai` | CLIP pretrained weights. `openai` trained on image-text pairs, well suited for real-world scene understanding. |
| `blip_model` | `blip-image-captioning-large` | BLIP captioning model. `large` slightly improves animal precision over `base` with ~1 min overhead. |

>[!TIP]
> See [Model Comparison](model-comparison.md) for benchmark results across different model combinations.

---

## FAKE_KEYS — Background Noise Descriptors

```json
"FAKE_KEYS": {
    "SHOE":   ["a photo of a shoe or footwear"],
    "WOOD":   ["a photo of a piece of wood or a stick"],
    "GARDEN": ["a photo of a garden object or debris"],
    "GROUND": ["a photo of the ground"]
}
```

Text descriptions used by CLIP to identify background noise and false positive sources. If a crop matches one of these descriptions with high similarity, it penalizes the classification score.

>[!NOTE]
> Per-camera weight for each fake key is controlled by `FAKE_WEIGHTS` in `cameras.json`. See [Camera Configuration](cameras-config.md).

---

## BLIP_KEYWORDS — Caption Matching

```json
"BLIP_KEYWORDS": {
    "PERSON":  ["person", "people", "human", "officer", "man", "woman", "child", "girl", "boy"],
    "VEHICLE": ["car", "truck", "bike", "motorcycle", "bus"],
    "ANIMAL":  ["cat", "dog", "horse", "bird", "bear"]
}
```

Keywords searched in the BLIP caption. If found, `BLIP_BOOST` is added to the category score.

**When to modify:**
- Add keywords for objects common in your region (e.g. `"scooter"` for VEHICLE)
- Add animal types specific to your area (e.g. `"fox"`, `"deer"`, `"rabbit"`)
- Remove keywords that cause false positives in your specific setup

---

## Score Weights

### FINAL_WEIGHT_CROP / FINAL_WEIGHT_FRAME

```json
"FINAL_WEIGHT_CROP":  0.7,
"FINAL_WEIGHT_FRAME": 0.3
```

How much the YOLO crop (zoomed detection) vs the full frame contributes to the final CLIP score. Must sum to 1.0.

- **Higher CROP weight** → relies more on the zoomed detection, better for small/distant objects
- **Higher FRAME weight** → relies more on scene context, better for large objects or ambiguous crops

### BLIP_BOOST

```json
"BLIP_BOOST": {
    "PERSON":  0.35,
    "ANIMAL":  0.10,
    "VEHICLE": 0.10
}
```

Score bonus added when BLIP caption contains a keyword for that category.

- **PERSON boost is intentionally higher** — the system prioritizes never missing a person (FN=0 goal)
- Lower animal/vehicle boost reflects that BLIP captions are less reliable for these categories

### FAKE_PENALTY_WEIGHT

```json
"FAKE_PENALTY_WEIGHT": {
    "PERSON":  0.3,
    "ANIMAL":  0.5,
    "VEHICLE": 0.6
}
```

How much fake/background scores penalize each category. Lower = category is more "protected" from false positive penalties.

- **PERSON=0.3** — minimal penalty, person detection is prioritized even if background is present
- **VEHICLE=0.6** — strongest penalty, vehicles need clean crops to be confirmed

---

## THRESHOLD — Minimum Score to Classify

```json
"THRESHOLD": {
    "PERSON":  0.15,
    "ANIMAL":  0.22,
    "VEHICLE": 0.30
}
```

Minimum final score required to classify a frame as that category. Scores below threshold are ignored.

| Value | Effect |
|-------|--------|
| Lower | More sensitive — fewer missed detections (FN↓), more false positives (FP↑) |
| Higher | More conservative — fewer false positives (FP↓), more missed detections (FN↑) |

- **PERSON threshold is intentionally low** (0.15) — better a false alarm than a missed person
- Can be overridden per-camera via `blip_rules.THRESHOLD` in `cameras.json`

---

## YOLO_NIGHT_BOOST

```json
"YOLO_NIGHT_BOOST": 0.30
```

Score bonus added to PERSON during nighttime (sunrise/sunset calculated from `city` in `settings.json`). CLIP is less accurate on dark/grainy night footage, so this compensates by giving extra weight to YOLO's person detection at night.

>[!NOTE]
> Increasing this value reduces night FN but may increase night FP on cameras with reflections or insects.

---

## BBOX_SMALL — Small Object Fix

```json
"BBOX_SMALL_RATIO":        0.04,
"BBOX_SMALL_PERSON_BONUS": 0.15
```

If a YOLO person detection occupies less than `BBOX_SMALL_RATIO` of the total frame area (e.g. a head peeking around a corner, a distant person), adds `BBOX_SMALL_PERSON_BONUS` to the PERSON score.

This fixes false negatives where a person is barely visible but YOLO correctly detected them.

- **BBOX_SMALL_RATIO=0.04** → activates for bounding boxes smaller than 4% of the frame
- Only applies when YOLO category is `person`

---

## AGGREGATION — Video-Level Decision

```json
"ANIMAL_START_THRESHOLD":  0.25,
"ANIMAL_STEP_REDUCTION":   0.05,
"ANIMAL_MIN_THRESHOLD":    0.15,

"VEHICLE_START_THRESHOLD": 0.50,
"VEHICLE_STEP_REDUCTION":  0.05,
"VEHICLE_MIN_THRESHOLD":   0.20
```

Controls how the per-frame scores are aggregated into a final video classification for ANIMAL and VEHICLE.

The threshold **starts high and decreases** with each positive frame found:

```
Frame 1 positive → threshold = START (0.25)
Frame 2 positive → threshold = START - STEP (0.20)
Frame 3 positive → threshold = START - 2*STEP (0.15) = MIN
```

More positive frames = lower threshold required = easier to classify the video.

**Tuning tips:**
- Many false positive animals → increase `ANIMAL_START_THRESHOLD`
- Missing too many animals → decrease `ANIMAL_MIN_THRESHOLD` or increase `ANIMAL_STEP_REDUCTION`
- VEHICLE too aggressive → increase `VEHICLE_START_THRESHOLD` (already higher than ANIMAL by default)

---

## Per-Camera Override

Any parameter above can be overridden per-camera in `config/cameras.json` under `blip_rules`:

```json
"blip_rules": {
    "FAKE_WEIGHTS": {
        "GROUND": 1.5,
        "GARDEN": 2.0
    },
    "THRESHOLD": {
        "PERSON": 0.10,
        "ANIMAL": 0.18
    },
    "BLIP_BOOST": {
        "PERSON": 0.40
    },
    "BBOX_SMALL_PERSON_BONUS": 0.20,
    "YOLO_NIGHT_BOOST": 0.40
}
```
>[!NOTE]
> Dict values (FAKE_WEIGHTS, THRESHOLD, BLIP_BOOST) are **merged** with global defaults — you only need to specify the keys you want to override.  
> Scalar values (BBOX_SMALL_PERSON_BONUS, YOLO_NIGHT_BOOST) **replace** the global value entirely.
