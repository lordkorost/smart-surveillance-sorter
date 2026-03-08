# 🔍 Lens Health Check

The Lens Health Check feature uses a Vision AI model (Ollama) to automatically detect dirty, obstructed, or obscured camera lenses by comparing a known-clean reference image with a recent night image from each camera.

> ℹ️ This feature requires **Ollama** and a Vision model (e.g. `qwen3-vl:8b`). See [AMD GPU Setup](gpu-setup-amd.md) for installation instructions.

---

## How It Works

For each camera:

1. Loads the **reference image** (a known-clean snapshot) from the `/checks` folder
2. Finds a **night image** from the NVR images associated with that camera — using astronomical night calculation based on your configured city coordinates (sunrise/sunset ± 20 minutes)
3. Sends **both images** to the Vision model with a prompt asking it to compare them and detect any new obstructions
4. Returns a verdict: `clean`, `dirty`, or `uncertain`

The Vision model looks for:
- White glowing filaments (spider webs)
- Blurry patches or dark spots not present in the reference
- Dust, moisture, or insects covering part of the lens

> ℹ️ Night images are used because IR illumination makes lens obstructions like spider webs clearly visible as bright glowing filaments.

---

## Setting Up Reference Images

Create a `/checks` folder in the project root and add one reference image per camera, named after the camera ID:

```
checks/
  00.jpg    ← reference for camera 00
  01.jpg    ← reference for camera 01
  02.png    ← .png is also supported
  ...
```

Supported formats: `.jpg`, `.jpeg`, `.png` (case-insensitive).

> ⚠️ Reference images should be taken when the lens is **known to be clean**. A night IR image works best — the same conditions as the comparison image.

> ℹ️ Cameras without a reference image in `/checks` are automatically skipped with a debug log message.

---

## Usage

### Standalone — check lenses only

```bash
sss --dir /path/to/nvr/footage --check-clean
```

No YOLO, no BLIP, no sorting — just builds the file index and runs the lens check. Fast and lightweight.

### Integrated — check lenses as part of a full scan

```bash
sss --dir /path/to/nvr/footage --refine --vision --check-clean
```

The lens check runs after the Vision refine step, using the same Ollama instance already loaded.

### From the WebUI

Go to **Tools → 🔍 Lens Health Check**, enter the input directory and optionally an output directory, then click **Run Lens Check**.

---

## Output

Results are saved to `lens_health.json` in the output directory:

```json
{
    "00": "clean",
    "01": "clean",
    "02": "dirty",
    "03": "uncertain"
}
```

A summary is also printed to the terminal:

```
────────────────────────────────────────
🔍 Lens Health Report:
  ✅ Camera 00 (Parking): CLEAN
  ✅ Camera 01 (Garden): CLEAN
  ⚠️  Camera 02 (Entrance): DIRTY
  ❓ Camera 03 (Orchard): UNCERTAIN
────────────────────────────────────────
```

Possible verdicts:

| Verdict | Meaning |
|---------|---------|
| `clean` | No obstructions detected — lens appears clear |
| `dirty` | Obstruction detected (spider web, dust, moisture, insect) |
| `uncertain` | Model could not determine lens status confidently |

---

## Example — Vision Model Reasoning

Below is an example of the Vision model's reasoning when detecting a dirty lens (spider web):

```
Thinking: Got it, let's analyze the images. First, check for any obstructions.
The reference image (first one) is the clean one. Compare with the current image
(second one).

Looking at the current image: there's a white glowing filament in the middle of
the tiled area. The reference image doesn't have that white streak. The current
image has a white blur or filament that's not in the reference. So that's an
obstruction. Therefore, the output should be 'dirty'.
```

The model compares the two images side by side, identifies elements present in the current image that were not in the reference, and determines whether they represent a lens obstruction.

---

## Customizing the Prompt

The prompt used for lens analysis can be customized per camera directly from the WebUI:

**WebUI → Cameras → Select camera → Vision Settings → Lens Check Prompt**

This allows you to add camera-specific context, for example:

- *"This camera faces a garden with an orange tree — ignore insects flying near the lens at night."*
- *"The right edge of this camera has a permanent shadow — ignore it."*

> ℹ️ If no custom prompt is set, the default prompt from `prompts_config.json` is used.

---

## Notes and Limitations

**IR insects (false positives):** At night with IR illumination active, flying insects near the lens appear as bright white dots or streaks. These can occasionally be misidentified as dirt. Adding a camera description mentioning IR night mode can help the model distinguish between insects in flight and persistent obstructions.

**No reference image:** If no reference image exists for a camera, that camera is skipped. The result will not appear in `lens_health.json`.

**No night images available:** If the input folder contains no NVR images taken during astronomical night for a given camera, the result is reported as `unknown`.

**Temperature:** The Vision model temperature for lens analysis defaults to `0.3` for consistent, deterministic results. Higher temperatures may cause the model to ignore camera descriptions.