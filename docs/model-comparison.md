# Model Comparison & Vision Temperature

**Dataset:** Folder March 1st — 521 videos + 480 images, 8 cameras, mixed scenes  
**Hardware:** AMD Ryzen 5 9600X | RX 9060 XT 16GB | ROCm 6.4 (Linux)\
**Cameras:** Reolink 4k outdoor

---

## CLIP Model Comparison

| CLIP Model | BLIP Model | BLIP Time | Total Time | Global Acc | Animal Precision | Animal Recall |
|------------|------------|-----------|------------|------------|-----------------|---------------|
| ViT-L-14 (openai) | base | 02:51 | ~46 min | 98.27% | 95.2% | 76.9% |
| ViT-L-14 (openai) | large | 04:12 | ~48 min | **98.46%** | **100.0%** | 76.9% |
| ViT-H-14 (laion) | large | 06:16 | ~50 min | 97.31% ❌ | 88.9% ❌ | 61.5% ❌ |

### Detailed Results

#### ViT-L-14 + BLIP base (default)
| Category | TP | FP | FN | Precision | Recall |
|----------|----|----|----|-----------|--------|
| PERSON | 188 | 8 | 0 | 95.9% | **100.0%** |
| VEHICLE | 11 | 0 | 1 | 100.0% | 91.7% |
| ANIMAL | 20 | 1 | 6 | 95.2% | 76.9% |
| OTHERS | 293 | 0 | 0 | 100.0% | 100.0% |
| **Global** | | | | **98.27%** | Avg: 89.53% |

#### ViT-L-14 + BLIP large ✅ Recommended
| Category | TP | FP | FN | Precision | Recall |
|----------|----|----|----|-----------|--------|
| PERSON | 188 | 8 | 0 | 95.9% | **100.0%** |
| VEHICLE | 11 | 0 | 1 | 100.0% | 91.7% |
| ANIMAL | 20 | 0 | 6 | **100.0%** | 76.9% |
| OTHERS | 294 | 0 | 0 | 100.0% | 100.0% |
| **Global** | | | | **98.46%** | Avg: 89.53% |

#### ViT-H-14 + BLIP large ❌ Not recommended
| Category | TP | FP | FN | Precision | Recall |
|----------|----|----|----|-----------|--------|
| PERSON | 188 | 9 | 0 | 95.4% | **100.0%** |
| VEHICLE | 11 | 0 | 1 | 100.0% | 91.7% |
| ANIMAL | 16 | 2 | 10 | 88.9% | 61.5% |
| OTHERS | 292 | 0 | 0 | 100.0% | 100.0% |
| **Global** | | | | **97.31%** | Avg: 84.40% |

### Conclusions
- **BLIP large** improves animal precision (removes 1 FP) with only ~1 min overhead → recommended
- **ViT-H-14** significantly worsens animal detection (-4 TP, +4 FN) despite being larger → bigger ≠ better for surveillance
- **ViT-L-14** remains the optimal CLIP model for this use case

---

## Vision Fallback Temperature Comparison

Tests run with `--refine --blip --fallback` on 22 ambiguous videos selected by the fallback logic.  
Models: ViT-L-14 + BLIP large

| Temperature | Fallback Time | Total Time | Global Acc | Person FP | Animal FN |
|-------------|--------------|------------|------------|-----------|-----------|
| 1.0 (default) | 02:51 (7.78s/vid) | ~51 min | 97.89% | 9 | 7 |
| 0.7 | 03:46 (10.31s/vid) | ~52 min | **98.08%** | **8** ✅ | 7 |
| 0.3 | 02:59 (8.16s/vid) | ~51 min | **98.08%** | **8** ✅ | 7 |
| 0.1 | ❌ infinite loop on 1 frame | — | — | — | — |

### Detailed Results

#### Temperature 1.0 (default)
| Category | TP | FP | FN | Precision | Recall |
|----------|----|----|----|-----------|--------|
| PERSON | 188 | 9 | 0 | 95.4% | 100.0% |
| VEHICLE | 11 | 0 | 1 | 100.0% | 91.7% |
| ANIMAL | 19 | 1 | 7 | 95.0% | 73.1% |
| OTHERS | 292 | 0 | 0 | 100.0% | 100.0% |
| **Global** | | | | **97.89%** | Avg: 88.25% |

#### Temperature 0.7
| Category | TP | FP | FN | Precision | Recall |
|----------|----|----|----|-----------|--------|
| PERSON | 188 | 8 | 0 | 95.9% | 100.0% |
| VEHICLE | 11 | 0 | 1 | 100.0% | 91.7% |
| ANIMAL | 19 | 1 | 7 | 95.0% | 73.1% |
| OTHERS | 293 | 0 | 0 | 100.0% | 100.0% |
| **Global** | | | | **98.08%** | Avg: 88.25% |

#### Temperature 0.3
| Category | TP | FP | FN | Precision | Recall |
|----------|----|----|----|-----------|--------|
| PERSON | 188 | 8 | 0 | 95.9% | 100.0% |
| VEHICLE | 11 | 0 | 1 | 100.0% | 91.7% |
| ANIMAL | 19 | 1 | 7 | 95.0% | 73.1% |
| OTHERS | 293 | 0 | 0 | 100.0% | 100.0% |
| **Global** | | | | **98.08%** | Avg: 88.25% |

#### Temperature 0.1 ❌
Model entered an infinite reasoning loop on an ambiguous night scene (black cat in low light).  
With `num_predict` unlimited, the model used all available tokens without producing a verdict.

### Conclusions
- **Temperature 0.7 and 0.3** improve results slightly over default 1.0 (1 fewer Person FP)
- **Temperature 0.3** achieves same results as 0.7 but slightly faster → recommended for fallback
- **Temperature 0.1** causes infinite loops on ambiguous scenes — never use
- Lowering temperature does not improve Animal recall — those FN are genuinely ambiguous scenes (black cat at night, tiny objects)

---

## Vision Mode (qwen3-vl:8b, temperature 1.0)

| Metric | Value |
|--------|-------|
| Speed | 1.83s/vid average |
| Total time | ~57 min (521 videos) |
| Global accuracy | 98.27% |
| Avg Recall | 91.92% |

| Category | TP | FP | FN | Precision | Recall |
|----------|----|----|----|-----------|--------|
| PERSON | 187 | 6 | 1* | 96.9% | 99.5% |
| VEHICLE | 11 | 0 | 1 | 100.0% | 91.7% |
| ANIMAL | 22 | 1 | 4 | 95.7% | 84.6% |
| OTHERS | 292 | 0 | 0 | 100.0% | 100.0% |

>[!NOTE]
>*FN on PERSON: arm visible through a window from inside — not considered a meaningful false negative.

>[!NOTE]
> Vision results are non-deterministic (temperature=1.0). FN on PERSON may be 0 or 1 depending on the reasoning path. In a second run: PERSON TP=188, FN=0.

---

## Key Takeaways

1. **ViT-L-14 + BLIP large** is the optimal model combination — best accuracy/speed tradeoff
2. **ViT-H-14 is not recommended** — larger model, worse results for surveillance
3. **Vision mode** achieves the best Animal recall (84.6% vs 76.9% BLIP) at the cost of longer processing
4. **BLIP+Fallback** does not reliably improve results — use only for cameras with specific false positive issues
5. **Temperature 0.3** is recommended for fallback mode — slightly better than default with no speed penalty
6. **0 missed persons (FN=0)** across all BLIP configurations — the system prioritizes person detection
