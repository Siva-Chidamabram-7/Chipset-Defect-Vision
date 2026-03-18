# Chipset Defect Vision — Change Log

All significant changes to the project, in reverse-chronological order.

---

## [Session 8] — 2026-03-18  SAM + YOLO Bootstrap Pipeline (Zero Manual Annotation)

### Overview
Implemented a fully automated training pipeline that requires zero manual
annotation.  SAM generates initial bounding-box labels from a class-labelled
folder structure.  YOLO trains on those labels.  YOLO then re-labels the
dataset with higher-quality predictions.  Re-training on the improved labels
closes the loop.

```
dataset/<ClassName>/    →  sam_auto_annotate  →  data/  →  train.py
                                                              ↓
                                                          best.pt
                                                              ↓
                                                    auto_label_with_yolo
                                                              ↓
                                                      improved data/  →  train.py
```

---

### `scripts/sam_auto_annotate.py` — new (Step 1)

**Purpose:** Convert a class-labelled folder dataset into YOLO-format train/val
splits using SAM for bounding-box generation.

**Input:**
```
dataset/
├── Missing_hole/
├── Mouse_bite/
├── Open_circuit/
├── Short/
├── Spur/
└── Spurious_copper/
```

**Output:**
```
data/
├── images/train/ + images/val/
└── labels/train/ + labels/val/
```

**Key design decisions:**

| Feature | Detail |
|---------|--------|
| Model type detection | Inferred from checkpoint filename (`vit_b` / `vit_l` / `vit_h`) |
| Per-class train/val split | 80/20 per class (default) for balanced representation |
| 5-stage mask filter | area → relative area → aspect ratio → border proximity → stability score |
| Greedy IoU-NMS | Removes duplicate overlapping SAM regions (threshold 0.50) |
| Fallback box | `0.5 0.5 1.0 1.0` (full image) when SAM finds no valid region — ensures every image has at least one label |
| `--overwrite` flag | Skip already-annotated images by default; `--overwrite` to redo |
| `--device` flag | Supports cpu / cuda / mps for SAM inference |

**Filtering constants:**
| Parameter | Default | Meaning |
|-----------|---------|---------|
| `min_area` | 500 px² | Minimum mask area |
| `max_area_ratio` | 0.60 | Maximum fraction of image area |
| min aspect ratio | 0.10 | Minimum w/h |
| max aspect ratio | 10.0 | Maximum w/h |
| border margin | 3 px | Edge exclusion zone |
| min stability | 0.75 | SAM stability score floor |
| NMS IoU threshold | 0.50 | Duplicate suppression |

**CLI:**
```bash
python scripts/sam_auto_annotate.py
python scripts/sam_auto_annotate.py --input dataset/ --output data/
python scripts/sam_auto_annotate.py --sam-weights weights/sam_vit_b.pth --device cuda
python scripts/sam_auto_annotate.py --val-split 0.2 --overwrite
```

---

### `scripts/auto_label_with_yolo.py` — new (Step 3)

**Purpose:** Replace SAM-generated labels with higher-quality YOLO predictions
after an initial training round.

**Behaviour:**
- Scans `data/images/train/` and `data/images/val/` for all images
- Runs YOLO inference with `--conf` threshold (default 0.30)
- Converts `xyxy` detections → normalised YOLO `cx cy w h`
- Writes new `.txt` labels to `data/labels/<split>/`, overwriting SAM labels
- `--keep-undetected`: preserve existing label when YOLO finds nothing
  (useful in early training iterations when the model is immature)

**CLI:**
```bash
python scripts/auto_label_with_yolo.py
python scripts/auto_label_with_yolo.py --weights weights/best.pt --conf 0.3
python scripts/auto_label_with_yolo.py --keep-undetected
```

---

### `requirements-sam.txt` — header updated
Comment updated: `scripts/generate_regions.py` → `scripts/sam_auto_annotate.py`
(generate_regions.py is deprecated; sam_auto_annotate.py is the active SAM script).

---

### Full bootstrap usage

```bash
# ── Install dependencies ──────────────────────────────────────────────────────
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-sam.txt      # adds segment-anything

# ── Step 1: SAM auto-annotation ───────────────────────────────────────────────
# Requires: dataset/<ClassName>/ layout + weights/sam_vit_h.pth (or vit_b/vit_l)
python scripts/sam_auto_annotate.py --input dataset/ --output data/

# ── Step 2: Initial YOLO training ─────────────────────────────────────────────
python !training/train.py --data !training/data.yaml --epochs 50

# ── Step 3: YOLO re-labelling ─────────────────────────────────────────────────
python scripts/auto_label_with_yolo.py --weights weights/best.pt --conf 0.3

# ── Step 4: Retrain on improved labels ────────────────────────────────────────
python !training/train.py --data !training/data.yaml --epochs 50

# ── Step 5: Repeat Steps 3–4 as desired ──────────────────────────────────────
```

---

## [Session 7] — 2026-03-18  6-Class Refactor + SAM Removal + Image Persistence

### Overview
Replaced binary good/defect model with a 6-class PCB defect model sourced from
Roboflow. Removed the SAM dataset pipeline entirely. Added binary decision layer
to the API. Added incoming image persistence.

---

### `!training/data.yaml` — 6-class update
| Field | Before | After |
|-------|--------|-------|
| `nc`  | `2`    | `6`   |
| names | `good`, `defect` | `Missing_hole`, `Mouse_bite`, `Open_circuit`, `Short`, `Spur`, `Spurious_copper` |
| Dataset source note | "local LabelImg" | "Roboflow YOLOv8 export" |

Path resolution unchanged (`path: ../data` relative to `!training/`).

---

### `!training/dataset.yaml` — updated to 6 classes
Mirror of `data.yaml`. Deprecation header retained, class names updated to match.

---

### `!training/prepare_dataset.py` — deprecated and disabled
**Reason:** Dataset is now exported directly from Roboflow in YOLOv8 format,
pre-split into train/val. Manual splitting is no longer required.

**Change:** Entire script body replaced with a deprecation stub that:
- Prints a clear error message with Roboflow export instructions
- Calls `sys.exit(1)` to prevent accidental execution

---

### `!training/train.py` — 3 targeted fixes
1. **`--data` default**: `"training/data.yaml"` → `"!training/data.yaml"`
   (matches the actual folder name on disk; previous value caused a silent
   path-not-found at runtime)
2. **`check_dataset()` YAML-missing error**: removed reference to
   `prepare_dataset.py`; now directs user to Roboflow export
3. **`check_dataset()` split-missing error**: replaced LabelImg + prepare_dataset
   instructions with Roboflow export instructions

No changes to hyperparameters, augmentation settings, or training logic.

---

### `app/model/predictor.py` — inference refactor

#### CLASS_NAMES — binary → 6-class
```python
# Before
CLASS_NAMES = {0: "good", 1: "defect"}

# After
CLASS_NAMES = {
    0: "Missing_hole",
    1: "Mouse_bite",
    2: "Open_circuit",
    3: "Short",
    4: "Spur",
    5: "Spurious_copper",
}
```

#### COLORS — updated to 6 distinct per-class colors
Each PCB defect class gets its own BGR color for clear visual annotation.
`"unknown"` grey retained for base-model fallback.

#### CONF_THRESHOLD — raised from 0.25 → 0.30
Reduces false positives in production.

#### `predict()` — simplified class resolution
- Removed COCO-class remap hack (`label = "defect" if conf < 0.6 else "good"`)
- Fine-tuned model: maps `cls` integer → `CLASS_NAMES` dict
- Base model: labels all detections `"unknown"` (COCO class IDs ≠ PCB defects)

#### Decision layer added
```python
status = "GOOD" if not detections else "DEFECT"
```

#### Response schema — breaking change
| Key | Before | After |
|-----|--------|-------|
| `status` | *(absent)* | `"GOOD"` or `"DEFECT"` |
| `detections[].label` | `"good"` / `"defect"` | class name string |
| `detections[].class` | *(absent)* | replaces `label` |
| `total` | int | **removed** |
| `defect_count` | int | **removed** |
| `good_count` | int | **removed** |
| `image` | base64 | base64 (unchanged) |
| `model` | string | string (unchanged) |

Full response shape:
```json
{
  "status":     "GOOD" | "DEFECT",
  "detections": [
    { "class": "Missing_hole", "confidence": 0.87, "bbox": [x1, y1, x2, y2] }
  ],
  "image": "<base64>",
  "model": "fine-tuned"
}
```

#### `_stub_response()` updated
Returns `"status": "GOOD"` and empty `detections` list matching new schema.

---

### `app/main.py` — image persistence + minor cleanup

#### New: `incoming_data/` storage
Every image posted to `/predict` is saved to `incoming_data/<unix_timestamp>.jpg`
for audit trails and future retraining data collection.

```python
INCOMING_DIR = Path(__file__).parent.parent / "incoming_data"
INCOMING_DIR.mkdir(parents=True, exist_ok=True)
# ...inside /predict:
save_path = INCOMING_DIR / f"{int(time.time())}.jpg"
save_path.write_bytes(img_bytes)
```

#### Refactored byte handling in `/predict`
Both `file` and `image_base64` paths now share a single `img_bytes` variable,
eliminating the `content` / `img_bytes` naming inconsistency in the original.

#### Version bumped: `1.0.0` → `2.0.0` (breaking API schema change)

---

### `scripts/generate_regions.py` — SAM removed and disabled
**Reason:** SAM (Segment Anything Model) pipeline is no longer part of the
dataset workflow. Dataset comes from Roboflow which provides pre-annotated,
pre-split data in YOLOv8 format.

**Change:** Entire SAM implementation replaced with a deprecation stub that:
- Prints a clear error with Roboflow instructions
- Calls `sys.exit(1)` to prevent accidental execution

All SAM imports (`segment_anything`, `SamAutomaticMaskGenerator`,
`sam_model_registry`) are gone from the codebase.

---

## [Session 6] — 2026-03-18  Docker Build Fix + Full Dockerignore Audit

### Bug fixed
| # | File | Problem | Fix |
|---|------|---------|-----|
| 1 | `Dockerfile` | `COPY training/ training/` tried to copy a directory excluded by `.dockerignore` → `CopyIgnoredFile` build error | Removed the COPY instruction entirely |
| 2 | `Dockerfile` | `COPY scripts/ scripts/` copied SAM + annotation tools into the inference image; dead weight | Removed; scripts are local-workstation-only tools |
| 3 | `.dockerignore` | Missing entries for `data/`, `raw_data/`, `scripts/`, `requirements-sam.txt`, IDE dirs | Added all missing exclusions with clear section comments |

### What the inference image now contains (and why)
```
app/          ← FastAPI application + predictor + utils
frontend/     ← glassmorphic HTML/CSS/JS UI (served by FastAPI /static)
weights/      ← best.pt (user fine-tuned, if present) + yolov8n.pt (baked in Layer 5)
requirements.txt  ← consumed in Layer 3, kept for pip reference
```

### Offline contract preserved
- Build-time: internet access OK (pip packages + yolov8n.pt weight download)
- Runtime: `--network none` safe; zero outbound calls; all assets baked in
- Env locks retained: `YOLO_TELEMETRY=False`, `YOLO_AUTOINSTALL=False`,
  `HF_DATASETS_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`

---

## [Session 5] — Dataset Pipeline Fixes + New Utility Scripts

### `scripts/annotate.py` — key-binding overhaul
| Key | Before | After |
|-----|--------|-------|
| Skip region | `S` | `X` |
| Next image  | *(missing)* | `N` — saves partial labels and jumps to next image |
| Undo | `B` | `B` (unchanged) |
| Quit | `Q` / `Esc` | `Q` / `Esc` (unchanged) |

Changed in: `KEY_SKIP` constant, `KEY_NEXT` constant (new), `draw_legend()` text,
`next_requested` state variable, key-handler branch, `status` return value
(`"quit"` / `"next"` / `"done"`), `main()` loop handler for `"next"` status,
startup hint string.

### `scripts/setup_data.py` — new migration utility
Moves images accidentally placed in `data/images/train|val` back to
`raw_data/images/` so the correct pipeline can run from Phase 1.
- Dry-run by default; `--execute` required to move files
- Handles filename collisions (`_2`, `_3`, … suffix)
- Prints four-step pipeline reminder on completion

### `scripts/check_pipeline.py` — new health checker
Phase-by-phase pipeline validator (Phases 0–6).
Reports `[OK]` / `[WARN]` / `[FAIL]` for:
- SAM checkpoint present in `weights/`
- `raw_data/images/` has images
- `raw_data/regions/` has JSON region files (+ JSON ↔ image cross-check)
- `raw_data/labels/` has label `.txt` files (+ label ↔ image cross-check)
- `data/images/train|val` populated (+ label alignment per split)
- `weights/yolov8n.pt` present
- `weights/best.pt` present
Exits with code 1 on any failure; `--verbose` for per-file mismatch details.

### `training/prepare_dataset.py` — pre-split validation guards
Three new checks run before the train/val split:
1. **Hard stop** — exits with actionable error if 0 images have any labels
2. **Warning** — printed if >50% of images are unannotated (likely an error)
3. **Orphan report** — lists label `.txt` files with no matching source image

---

## [Session 4] — Hybrid SAM + YOLO Pipeline

### New files
- `scripts/generate_regions.py` — Phase 1: SAM automatic mask generation
  - Loads SAM from local checkpoint only; hard-fail if missing
  - 5-stage filter: area → aspect ratio → border proximity → IoU-NMS → quality score
  - Outputs per-image JSON to `raw_data/regions/`
  - `--overwrite` flag; skips cached JSONs by default
- `scripts/annotate.py` — Phase 2: interactive OpenCV annotation tool
  - Two windows: full PCB overview + zoomed region crop
  - Saves YOLO `.txt` to `raw_data/labels/` on image completion
  - Resume support (skips images that already have label files)
  - Headless-display guard (exits with clear message if no display)
- `requirements-sam.txt` — SAM + GUI opencv deps (dataset creation only)

### Architecture decision
SAM is strictly a **dataset-creation** tool. It is never loaded at inference time.
This separation keeps the inference Docker image at ~500 MB instead of ~3 GB.

### `.gitignore` — SAM intermediate outputs
Added `raw_data/regions/**/*.json` to prevent large JSON files from being committed.

---

## [Session 3] — Offline-First Hardening

### `app/model/predictor.py` — removed auto-download fallback
Removed the Priority-3 `YOLO("yolov8n.pt")` call that silently triggered a
network download in air-gapped environments.
New behaviour: missing weights → logged ERROR to stderr + stub mode (visible
banner overlay on every prediction image).

### `Dockerfile` — offline environment locks
Added `ENV` block:
```
YOLO_CONFIG_DIR=/app/.yolo
YOLO_TELEMETRY=False
YOLO_AUTOINSTALL=False
HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```
Layer 5 bakes `yolov8n.pt` into the image at build time with a double assertion:
Python `sys.exit(1)` + shell `test -f` — both must pass for the build to succeed.

### Non-root user fix
`adduser --home /app` ensures the ultralytics config fallback (`~/.config/Ultralytics`)
resolves to `/app/.yolo` (overridden by `YOLO_CONFIG_DIR`) instead of `/root`
which is inaccessible to non-root users.

---

## [Session 2] — Offline LabelImg Workflow + Dataset Structure

### Removed
- Roboflow integration (all `roboflow` imports and API key references)

### `training/data.yaml`
```yaml
path: ../data
train: images/train
val:   images/val
nc: 2
names: {0: good, 1: defect}
```

### `training/train.py` — offline pre-flight guards
- `check_model_weights()`: resolves bare filename to `weights/<name>`;
  exits with download instructions if file missing
- `check_dataset()`: verifies YAML, split dirs, non-empty images;
  exits with actionable fix steps if broken
- `--device` default changed from `"cuda"` to `"cpu"`
- `--data` default: `"training/data.yaml"`

### `training/prepare_dataset.py` — defaults updated
- `--src` default: `"raw_data"` (was hard-coded or missing)
- `--dst` default: `"data"`
- Two-way 80/20 split (removed test split)
- Creates empty `.txt` for images without labels (background examples)

### Directory scaffolding
`.gitkeep` placeholder files in all required empty directories:
`data/images/train|val`, `data/labels/train|val`,
`raw_data/images`, `raw_data/labels`, `raw_data/regions`

### `.gitignore` — fixed negation failure
Replaced directory-level exclusion (`data/images/`) with extension-level
exclusion (`data/images/**/*.jpg`) so `.gitkeep` placeholder files are still
tracked by git while dataset images are ignored.

---

## [Session 1] — Initial Build + Dependency Fixes

### Project scaffolded
Complete system built from scratch:
- `app/main.py` — FastAPI application with `/predict`, `/health`, static UI
- `app/model/predictor.py` — YOLOv8 wrapper with class-colour annotation
- `app/utils/image_utils.py` — base64 decode + image validation helpers
- `frontend/index.html` — glassmorphic vanilla-JS drag-and-drop UI
- `Dockerfile` — single-stage offline-first build
- `requirements.txt` — pinned inference dependencies

### Dependency conflict fixes
| Conflict | Fix |
|----------|-----|
| `pydantic` v1/v2 mismatch (langchain vs FastAPI) | Removed langchain; pinned `pydantic==2.7.1` |
| `opencv-python` (GUI) vs `opencv-python-headless` | Install ultralytics first, then `pip uninstall opencv-python && pip install opencv-python-headless==4.9.0.80` |

### Dockerfile: invalid `COPY … \|\| true` syntax
Docker `COPY` is not a shell command; `|| true` is invalid.
Fixed by removing the multi-stage approach and using a single-stage build with
a Python `sys.exit()` assertion instead.

### Cloud Run compatibility
`PORT` env var respected; `CMD` uses `--host 0.0.0.0 --port 8080`.
