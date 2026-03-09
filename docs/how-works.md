# How It Works

## The Problem

NVR systems have limited storage depending on the hard drive size. When set to record 24/7, they eventually run out of space and delete older footage. In case of failure — which happened to me twice — they simply stop recording, leaving no evidence of an intrusion.

My solution was to configure the NVR to record only motion events and upload them to my home server via FTP. However, wind, spider webs, rain, and other triggers — despite sensitivity settings and motion masks — still generate many "useless" videos. Reviewing all of them manually is impractical.

**Smart Surveillance Sorter** solves this: it categorizes the videos and extracts key frames so you get an immediate visual of what each video contains. Instead of watching 400+ videos, you review a handful to confirm nothing important was missed.

---

## The Idea

The goal is to process videos without watching them all, and discard irrelevant ones before filling up a terabyte of FTP recordings — and before an NVR hard drive failure destroys evidence of a real intrusion.

The system must be **fast enough to be practical** and **highly configurable**: if a camera is replaced, the NVR changes, or indoor cameras are added, the system should continue working correctly. This is made possible by per-camera configuration of almost every parameter — similar to how NVRs allow per-camera sensitivity and trigger settings.

>[!TIP]
> The NVR configuration matters just as much as the sorter settings. The more "false" videos the NVR records, the more the sorter has to process — increasing both processing time and false positives.

---

## The Pipeline

Two pipelines were designed: one **CPU-friendly** for machines without a dedicated GPU, and one **more accurate but GPU-dependent** for acceptable processing times. Both use YOLO as the first step.

**Step 1 — YOLO on NVR images**  
YOLO scans any NVR snapshot images associated with a video. If an image contains a person with high confidence, that video is immediately categorized as PERSON. This step takes only seconds and eliminates those videos from further analysis.

**Step 2 — YOLO on videos**  
Videos not categorized in Step 1 are scanned frame by frame. Analysis speed depends on per-camera settings and the actual video FPS. For example, with `stride=0.6s`, one frame is analyzed every 0.6 seconds, skipping `stride × FPS` frames between each sample. Detected objects (person, animal, vehicle) are saved as frames and crops according to the active mode and per-camera `ignore_labels`.

The pipeline then splits into two options:

**Option A — Fast (YOLO + BLIP/CLIP)**  
Frames and crops saved by YOLO are processed by BLIP and CLIP, which determine the final video category. Optionally, a fallback step sends uncertain videos — where YOLO and BLIP/CLIP disagree — to a Vision model for a final decision on those few remaining cases.

**Option B — Accurate (YOLO + Vision)**  
Frames saved by YOLO are sent directly to a Vision model via Ollama (`qwen3-vl:8b` has proven the most reliable), which analyzes the full 4K frames and determines the video category.

>[!NOTE]
> In both pipelines, if YOLO detects a person with medium-high confidence but BLIP/CLIP or Vision disagrees, **YOLO wins**. The refinement steps primarily serve to reduce false positives, not to override confident person detections.

---

## Why a Chain?

The order of steps is not arbitrary — everything traces back to NVR configuration quality.

- **YOLO on NVR images** — eliminates videos with clear persons in seconds
- **YOLO on videos** — the most time-consuming step; fewer "false" NVR recordings = fewer videos to scan
- **BLIP/CLIP** — fast refinement, depends on the number of frames saved by YOLO
- **Vision** — slower refinement, also depends on the number of frames saved by YOLO

A well-configured NVR makes every subsequent step faster and more accurate.

---

## Subject-Based Detection vs Motion Detection

An alternative approach would have been classic motion detection — flag any video where pixels change. However, this was deliberately avoided.

Motion detection catches everything: wind, rain, spider webs, headlights, shadows. The NVR already does this and still generates hundreds of false recordings.

The key insight is that **everything we care about has a subject**:
- **Persons** — move on their own ✅
- **Animals** — move on their own ✅  
- **Vehicles** — only move when driven by a person → detecting the person while getting in or out a vehicle is equivalent to detecting a new vehicle event ✅
- **Wind, rain, spider webs** — not subjects → YOLO ignores them with the right confidence settings ✅

This is why vehicles can be safely ignored on a parking camera — any car entering or leaving must have a driver, who will be detected getting in or out. The same logic applies to a barn camera: ignore the livestock that are always there, detect the unexpected subject (person, predator) that represents a real event.

SSS doesn't detect movement — it detects **intent**.

> See [Mode Comparison](benchmarks/mode-comparison.md) for benchmark results across different modes and ignore_labels configurations.  
> See [Camera Configuration](cameras-config.md) for per-camera ignore_labels setup and examples.

---

## Person First — Always

In a security context, **persons have absolute priority** — even at the cost of some false positives.

Every step of the system is tuned to favor person detection: lower thresholds, higher score boosts, YOLO override on medium-high confidence detections. This results in near-zero false negatives on persons — across all tests, approximately 10 missed persons out of 7,000+ videos analyzed, after per-camera tuning.

Those 10 were still recoverable from other cameras covering the same area. In a well-designed surveillance system, cameras should overlap — what one camera misses, the adjacent camera covering the same scene from a different angle will catch.

---

## Early Exit

Beyond configurable stride, YOLO uses an **early exit mechanism** to avoid scanning entire videos when enough evidence is already found.

Given `num_occurrence` and `time_gap_sec`, YOLO stops analyzing a video and moves to the next one as soon as the same category is detected `num_occurrence` times, each at least `time_gap_sec` apart.

**Example** with `num_occurrence=3`, `time_gap_sec=3` on a 1-minute video:
- Person detected at 0:05 → occurrence 1
- Person detected at 0:08 → occurrence 2 (+3s gap ✅)
- Person detected at 0:11 → occurrence 3 (+3s gap ✅) → **early exit**

Only 11 seconds analyzed out of 60. The same logic applies to animals.

**Early exit does not apply to vehicles.** The reasoning:

- If a **person** is confirmed → safe to stop, the video is categorized
- If an **animal** is confirmed → a domestic animal usually has an owner nearby (who will also be detected); a stray animal will likely flee if a person approaches (calculated risk)
- If a **vehicle** is confirmed → a person may exit the vehicle at any moment. Stopping early on a vehicle detection would risk missing that person — unacceptable

Empty videos (no detections) are always scanned in full — just as a human reviewer would have to watch them entirely to confirm they're empty. This is another reason why minimizing false NVR triggers directly reduces processing time.

---

## YOLO vs BLIP/CLIP vs Vision

**YOLO (yolov8l)**  
With the right minimum confidence settings, highly reliable for person detection — even distant silhouettes, low light, partial occlusion. However, it tends to generate false positives on shadows, rain, insects, and reflections even at medium-high confidence.

**BLIP + CLIP**  
With the right fake/background penalty configuration, effectively reduces YOLO false positives on persons. Less reliable on animals — it struggles to distinguish animals from background objects in ambiguous crops.

**Vision (qwen3-vl:8b)**  
With the right prompt, acts as a true scene descriptor — analyzes full 4K frames and rarely misses anything visible. Highly reliable on animals, significantly reduces both person false positives and animal false negatives. 

>[!NOTE]
>Limitations: at `temperature=1.0` the model is non-deterministic — on genuinely ambiguous scenes (very distant, dark, small subjects) it may produce inconsistent results between runs. On rare occasions, a person that YOLO detected with low confidence may be classified as "nothing" by Vision if they are too small and dark in the 4K frame. The YOLO override clause covers most of these cases.

---

## Known Limitations

- **Fast-moving small animals at night** (e.g. black cat) may not be detected by YOLO, or may be discarded by BLIP/CLIP after detection
- **Running persons** may be missed if `vid_stride_sec` is set too high — with default `0.6s` this is rarely an issue
- **Partially visible persons** (head at the bottom corner, arm through a window or car windshield) are sometimes detected by YOLO, sometimes not — not considered a security risk with overlapping camera coverage
- **Large animals misclassified as persons** — the system is tuned to prioritize persons, so if YOLO or BLIP classifies an animal as a person in one step, the video may end up in the PERSON folder. Vision mode significantly reduces this
- **Spider web on camera lens** — triggers detections on every video until cleaned. Use `FAKE_KEYS` or clean the lens
