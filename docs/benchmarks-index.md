# 📊 Benchmarks

This section contains detailed benchmark results for Smart Surveillance Sorter across different configurations, hardware, and datasets.

---

## 🗂️ Test Datasets

| Dataset | Videos | Images | Duration | Description |
|---------|--------|--------|----------|-------------|
| [Folder March 1st](benchmarks/folder-march-1st.md) | 521 | 480 | 1 day | 8 cameras, mixed scenes, day/night |
| [Model Comparison](benchmarks/model-comparison.md) | 521 | 480 | 1 day | Same dataset, different CLIP/BLIP models and Vision temperatures |
| [YOLO Parameter Tuning](benchmarks/yolo-tuning.md) *(coming soon)* | 399 | — | 1 day | Mostly empty footage (6+ hour), rain, parameter tuning |
| [Edge Cases](benchmarks/edge-cases.md) *(coming soon)* | — | — | — | Wood/bird false positives, ignore_labels |

---

## 🔬 How Tests Were Run

All tests were run with the `--test --no-sort` flags to avoid modifying the original footage:

```bash
sss --dir /path/to/dataset --mode full --refine --blip --test --no-sort
sss --dir /path/to/dataset --mode full --refine --vision --test --no-sort
```

Ground truth was generated manually using:
```bash
sss --dir /path/to/dataset --ground
```

Results were compared using:
```bash
sss --dir /path/to/dataset --compare
```

---

## 📐 Metrics Explained

- **Precision** — of all videos classified as category X, how many were actually X. High precision = few false alarms.
- **Recall** — of all real X videos, how many were correctly detected. High recall = few missed detections.
- **Global accuracy** — percentage of correctly classified videos overall.
- **Avg Recall (excl. Others)** — average recall across PERSON, ANIMAL, VEHICLE only.

> 🎯 For a surveillance system, **Recall on PERSON is the most critical metric** — a missed person (FN=0) is always worse than a false alarm.

---

## 🖥️ Test Hardware

| Component | Spec |
|-----------|------|
| CPU | AMD Ryzen 5 9600X |
| GPU | AMD RX 9060 XT 16GB |
| RAM | 32GB DDR5 |
| OS (Linux) | Ubuntu 24.04 + ROCm 6.4 |
| OS (Windows) | Windows 11 + ROCm 7.2 + Vulkan |
| YOLO VRAM | ~8% |
| BLIP+CLIP VRAM | ~29% |
| Vision VRAM | ~57% |


**Test cameras:** Reolink 4K
- Daytime: 20 fps
- Nighttime: 12 fps
- Resolution: 3840×2160

> ℹ️ Parameters tuned for Reolink 4K footage. Other cameras with similar specs (4K, 12-20fps) should work well with default parameters. Lower resolution or fps cameras may need stride/occurrence adjustments — see [YOLO Tuning](docs/benchmarks/yolo-tuning.md).
