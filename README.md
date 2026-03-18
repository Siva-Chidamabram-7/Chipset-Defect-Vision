# Chipset Defect Vision

**YOLOv8 PCB defect detection** — 6 defect classes · SAM bootstrap · FastAPI · Glassmorphic UI · Docker · Fully offline

---

## What it does

Detects and classifies six types of PCB manufacturing defects in real-time using YOLOv8.
Every prediction returns a binary pass/fail decision plus per-defect bounding boxes.

| Class ID | Defect Class | Description |
|---|---|---|
| 0 | `Missing_hole` | Drilled hole absent or blocked |
| 1 | `Mouse_bite` | Irregular edge bite on board outline |
| 2 | `Open_circuit` | Broken trace / incomplete connection |
| 3 | `Short` | Unintended conductive bridge |
| 4 | `Spur` | Unwanted copper protrusion |
| 5 | `Spurious_copper` | Excess copper in non-conductive area |

---

## Feature Overview

| Feature | Detail |
|---|---|
| **6-class detection** | All six standard PCB defect types |
| **Binary verdict** | Every prediction returns `"GOOD"` or `"DEFECT"` |
| **Zero-annotation pipeline** | SAM auto-labels; YOLO self-improves — no manual labeling needed |
| **Offline-first** | Zero runtime network calls; all assets baked in at build time |
| **Image persistence** | Every incoming image saved to `incoming_data/` for audit + retraining |
| **Docker ready** | Single-stage CPU build, non-root runtime, hard-fail weight verification |
| **Glassmorphic UI** | Drag-and-drop upload + live camera capture; zero JS frameworks |
| **Health endpoint** | `GET /health` for container orchestration / load balancer probes |

---

## Architecture

### Inference flow (production / Docker)

```
POST /predict
    │
    ▼  app/main.py
    ├── save image  →  incoming_data/<timestamp>.jpg
    │
    ▼  app/model/predictor.py
    ├── weights/best.pt      (fine-tuned — loaded first if present)
    └── weights/yolov8n.pt   (base COCO model baked in at Docker build time)
         │
         ▼  YOLOv8 inference
         │
         ├── detections: [{class, confidence, bbox}, …]
         └── status: "GOOD" (no detections) | "DEFECT" (any detection)
```

### Dataset pipeline (local workstation — offline)

```
dataset/
├── Missing_hole/   ← raw images organised by class
├── Mouse_bite/
├── Open_circuit/
├── Short/
├── Spur/
└── Spurious_copper/
        │
        ▼  Step 1 — python scripts/sam_auto_annotate.py
        │           SAM auto-generates bounding boxes
        │           class ID assigned from folder name
        │           80/20 per-class train/val split
        │
data/images/train|val  +  data/labels/train|val   (YOLO format)
        │
        ▼  Step 2 — python !training/train.py
        │           YOLOv8 fine-tuning on SAM labels
        │
weights/best.pt
        │
        ▼  Step 3 — python scripts/auto_label_with_yolo.py
        │           Replace SAM labels with sharper YOLO predictions
        │
        ▼  Step 4 — python !training/train.py   (retrain)
        │
        └── Repeat Steps 3–4 as many times as desired
```

> **SAM is only used during dataset creation.**
> **YOLO is the only model used at inference time.**
> They are never loaded simultaneously.

---

## Project Structure

```
Chipset-Defect-Vision/
├── app/
│   ├── main.py                    ← FastAPI — /predict, /health
│   ├── model/
│   │   └── predictor.py           ← YOLOv8 wrapper, binary decision layer
│   └── utils/
│       └── image_utils.py         ← base64 decode, image validation
│
├── !training/
│   ├── data.yaml                  ← YOLO dataset config (6 classes)
│   ├── dataset.yaml               ← mirror of data.yaml
│   ├── train.py                   ← offline fine-tuning with pre-flight checks
│   └── prepare_dataset.py         ← DEPRECATED — exits with error if run
│
├── scripts/
│   ├── sam_auto_annotate.py       ← Step 1: SAM → YOLO labels (zero annotation)
│   ├── auto_label_with_yolo.py    ← Step 3: YOLO re-labels dataset (bootstrap)
│   ├── check_pipeline.py          ← health checker — validates all pipeline phases
│   ├── setup_data.py              ← migration: move images to correct directory
│   ├── annotate.py                ← legacy interactive OpenCV labeling tool
│   └── generate_regions.py        ← DEPRECATED — exits with error if run
│
├── data/                          ← YOLO split dataset (git-ignored content)
│   ├── images/train/ + val/
│   └── labels/train/ + val/
│
├── dataset/                       ← source images, one folder per class
│   ├── Missing_hole/
│   ├── Mouse_bite/
│   ├── Open_circuit/
│   ├── Short/
│   ├── Spur/
│   └── Spurious_copper/
│
├── weights/                       ← model checkpoints (git-ignored)
│   ├── yolov8n.pt                 ← base weights (baked into Docker image)
│   ├── best.pt                    ← fine-tuned model (produced by train.py)
│   ├── sam_vit_b.pth              ← SAM ViT-B (download once, use offline)
│   ├── sam_vit_l.pth              ← SAM ViT-L (optional)
│   └── sam_vit_h.pth              ← SAM ViT-H (optional, best quality)
│
├── incoming_data/                 ← images received by /predict (auto-created)
├── frontend/
│   ├── index.html                 ← glassmorphic drag-and-drop + camera UI
│   ├── styles.css
│   └── script.js
│
├── Dockerfile                     ← single-stage CPU build, inference only
├── .dockerignore                  ← excludes training/, scripts/, data/, raw_data/
├── requirements.txt               ← inference dependencies (FastAPI + ultralytics)
├── requirements-sam.txt           ← dataset-creation dependencies (SAM + opencv)
└── CHANGELOG.md
```

---

## Quick Start

### Option A — Docker (recommended for production)

```bash
# Build once (internet required — downloads wheels + yolov8n.pt)
docker build -t chipset-defect-vision .

# Run (fully offline)
docker run --network none -p 8080:8080 chipset-defect-vision
```

Open http://localhost:8080

---

### Option B — Local development

```bash
# Create inference venv
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux

pip install torch==2.3.0 torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Run inference server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

Open http://localhost:8080

---

## Dataset Preparation — Zero-Annotation Bootstrap

### Prerequisites

```bash
# Separate venv for dataset work (SAM + GUI opencv)
python -m venv .venv-sam
.venv-sam\Scripts\activate        # Windows
source .venv-sam/bin/activate     # macOS / Linux

pip install torch==2.3.0 torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-sam.txt
```

---

### Step 0 — Organise your images

Place PCB images in a class-labelled folder structure:

```
dataset/
├── Missing_hole/     ← all photos of Missing_hole defects
├── Mouse_bite/
├── Open_circuit/
├── Short/
├── Spur/
└── Spurious_copper/
```

Photography tips:
- Consistent overhead lighting, no harsh shadows
- Resolution ≥ 640 × 640 px
- One defect region clearly visible per image (or at least dominant)

---

### Step 0b — Download SAM checkpoint (once, on a connected machine)

```bash
# ViT-B — recommended (375 MB, good for CPU)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
     -O weights/sam_vit_b.pth

# ViT-L — better accuracy (1.2 GB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth \
     -O weights/sam_vit_l.pth

# ViT-H — best accuracy (2.4 GB, GPU recommended)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
     -O weights/sam_vit_h.pth
```

Transfer the checkpoint to `weights/` on your offline workstation.
The script auto-detects the model type from the filename (`vit_b` / `vit_l` / `vit_h`).

---

### Step 1 — SAM auto-annotation

```bash
# Default (dataset/ → data/, uses weights/sam_vit_h.pth, CPU)
python scripts/sam_auto_annotate.py

# Explicit options
python scripts/sam_auto_annotate.py \
    --input       dataset/ \
    --output      data/ \
    --sam-weights weights/sam_vit_b.pth \
    --device      cpu \
    --val-split   0.20 \
    --overwrite
```

**What it does per image:**
1. Load image from `dataset/<ClassName>/`
2. Run SAM automatic mask generator
3. Filter masks through 5-stage pipeline:

| Stage | Rule | Default |
|---|---|---|
| Minimum area | Drop masks < N pixels² | 500 px² |
| Maximum area | Drop masks covering > N% of image | 60% |
| Aspect ratio | Drop masks outside w/h range | 0.10 – 10.0 |
| Border proximity | Drop masks touching image edge | 3 px margin |
| Stability score | Drop low-quality SAM masks | ≥ 0.75 |
| IoU-NMS | Remove duplicate overlapping masks | threshold 0.50 |

4. **Fallback:** if no masks survive, write a full-image box (`0.5 0.5 1.0 1.0`)
   so every image has at least one label for YOLO
5. Assign `class_id` from folder name
6. Write YOLO `.txt` label to `data/labels/<split>/`
7. Copy image to `data/images/<split>/`

**Output:**
```
data/
├── images/train/   ← 80% of each class
├── images/val/     ← 20% of each class
├── labels/train/   ← YOLO-format .txt
└── labels/val/
```

---

### Step 2 — Initial YOLO training

```bash
# CPU (slower, any machine)
python !training/train.py --epochs 50 --imgsz 640 --device cpu

# GPU (faster)
python !training/train.py --epochs 50 --imgsz 640 --batch 16 --device 0
```

Pre-flight checks run automatically before training:
- Verifies `weights/yolov8n.pt` exists locally (no silent downloads)
- Verifies `data/images/train/` and `val/` are non-empty
- Exits with actionable error messages if anything is missing

Best weights are copied to `weights/best.pt` automatically.

---

### Step 3 — YOLO re-labelling (bootstrap improvement)

```bash
python scripts/auto_label_with_yolo.py

# Options
python scripts/auto_label_with_yolo.py \
    --weights weights/best.pt \
    --conf    0.30 \
    --device  cpu \
    --keep-undetected    # keep old label when YOLO finds nothing
```

Scans all images in `data/images/train/` and `val/`, runs YOLO inference,
replaces the SAM labels in `data/labels/` with YOLO predictions.

**`--keep-undetected`:** In early iterations the model may miss some defects.
This flag preserves the existing SAM label rather than overwriting with empty —
useful for the first one or two bootstrap rounds.

---

### Step 4 — Retrain on improved labels

```bash
python !training/train.py --epochs 50 --imgsz 640
```

Repeat Steps 3–4 as many times as desired. Each round typically improves
precision and recall as YOLO replaces coarse SAM proposals with tighter boxes.

---

### Full bootstrap in one block

```bash
# Step 1 — SAM labels
python scripts/sam_auto_annotate.py --input dataset/ --output data/

# Step 2 — Initial model
python !training/train.py --data !training/data.yaml --epochs 50

# Step 3+4 — First improvement round
python scripts/auto_label_with_yolo.py --conf 0.3
python !training/train.py --data !training/data.yaml --epochs 50

# Step 3+4 — Second improvement round (optional)
python scripts/auto_label_with_yolo.py --conf 0.35
python !training/train.py --data !training/data.yaml --epochs 50
```

---

## API Reference

### `GET /health`

```bash
curl http://localhost:8080/health
```

```json
{
  "status":       "ok",
  "model_loaded": true,
  "version":      "2.0.0"
}
```

---

### `POST /predict`

Accepts either a multipart file upload or a base64-encoded image string.
Every incoming image is also saved to `incoming_data/<unix_timestamp>.jpg`
for audit trail and future retraining data collection.

#### Upload a file

```bash
curl -X POST http://localhost:8080/predict \
     -F "file=@pcb_sample.jpg"
```

#### Send base64

```bash
B64=$(base64 -i pcb_sample.jpg)
curl -X POST http://localhost:8080/predict \
     -F "image_base64=${B64}"
```

#### Response schema

```json
{
  "status": "GOOD",
  "detections": [],
  "image": "<base64-encoded annotated JPEG>",
  "model": "fine-tuned"
}
```

```json
{
  "status": "DEFECT",
  "detections": [
    {
      "class":      "Missing_hole",
      "confidence": 0.9143,
      "bbox":       [120, 45, 210, 130]
    },
    {
      "class":      "Spur",
      "confidence": 0.7821,
      "bbox":       [310, 180, 370, 240]
    }
  ],
  "image": "<base64-encoded annotated JPEG>",
  "model": "fine-tuned"
}
```

**Decision rule:**
- `"GOOD"` — no detections above confidence threshold (0.30)
- `"DEFECT"` — one or more detections above threshold

**`model` field values:**
| Value | Meaning |
|---|---|
| `"fine-tuned"` | `weights/best.pt` loaded — 6-class PCB model |
| `"base-yolov8n"` | No `best.pt`; using base COCO model (detections labelled `"unknown"`) |
| `"stub — no weights found"` | Neither weight file present — server still responds, inference disabled |

---

## Camera Usage

1. Open http://localhost:8080
2. Click the **Live Camera** tab
3. Click **Start Camera** → allow browser permission
4. Point camera at PCB board
5. Click **Capture Frame** then **Scan Image**

> Camera uses `navigator.mediaDevices.getUserMedia`.
> Must be served over **HTTPS** or `localhost`.

---

## Docker

### Build (internet required — one-time)

```bash
docker build -t chipset-defect-vision .
```

The build performs these layers in order:

| Layer | What happens |
|---|---|
| 1 | System libs (`libglib2.0`, `libgomp1`, `libgl1`) |
| 2 | PyTorch 2.3.0 CPU-only wheel |
| 3 | Python deps from `requirements.txt`; swap `opencv-python` → `opencv-python-headless` |
| 4 | *(optional, commented)* `segment-anything` — uncomment if running SAM in Docker |
| 5 | Download `yolov8n.pt` into `/app/weights/` — **build fails** if download fails |
| 6 | Copy `app/` + `frontend/` + `weights/` |
| 7 | Create non-root `appuser` and set permissions |

**What is intentionally NOT copied into the image:**

| Excluded | Reason |
|---|---|
| `training/` / `!training/` | Training runs locally, not in the inference container |
| `scripts/` | SAM + annotation tools need a display and heavy deps |
| `data/`, `raw_data/` | Dataset directories — can be gigabytes |

### Run (fully offline)

```bash
# Standard
docker run -p 8080:8080 chipset-defect-vision

# Prove zero internet required
docker run --network none -p 8080:8080 chipset-defect-vision
```

### Mount fine-tuned weights

```bash
docker run --network none -p 8080:8080 \
    -v "$(pwd)/weights/best.pt:/app/weights/best.pt:ro" \
    chipset-defect-vision
```

### Transfer to offline machine

```bash
# Connected machine — save image to file
docker save chipset-defect-vision | gzip > chipset-defect-vision.tar.gz

# Offline machine — load and run
docker load < chipset-defect-vision.tar.gz
docker run --network none -p 8080:8080 chipset-defect-vision
```

---

## Offline Guarantee

| Stage | Internet required? | What happens |
|---|---|---|
| Download SAM checkpoint | ✅ Once | `wget` → transfer to `weights/` |
| `pip install -r requirements-sam.txt` | ✅ Once | Install on dataset workstation |
| `docker build` | ✅ Once | Downloads deps + bakes `yolov8n.pt` |
| SAM annotation + YOLO training | ❌ None | Fully local |
| `docker run` (inference) | ❌ None | Loads weights from disk; zero outbound calls |

**Offline locks set in the Docker image (`ENV`):**

| Variable | Value | Effect |
|---|---|---|
| `YOLO_CONFIG_DIR` | `/app/.yolo` | Redirects config writes away from `/root` |
| `YOLO_TELEMETRY` | `False` | No outbound analytics |
| `YOLO_AUTOINSTALL` | `False` | No auto-package installs |
| `HF_DATASETS_OFFLINE` | `1` | HuggingFace offline mode |
| `TRANSFORMERS_OFFLINE` | `1` | Transformers offline mode |

---

## Utility Scripts

### `scripts/check_pipeline.py` — pipeline health checker

Validates every phase before you run it.

```bash
python scripts/check_pipeline.py
python scripts/check_pipeline.py --verbose   # per-file mismatch details
```

Checks phases 0–6:

| Phase | Checks |
|---|---|
| 0 | SAM checkpoint present in `weights/` |
| 1 | `dataset/<ClassName>/` folders exist and contain images |
| 2 | `data/images/train\|val/` non-empty |
| 3 | `data/labels/train\|val/` non-empty; label ↔ image alignment |
| 4 | Train/val split sanity; orphan labels |
| 5 | `weights/yolov8n.pt` present |
| 6 | `weights/best.pt` present |

Exits with code `1` on any failure, `0` when everything is ready.

---

### `scripts/setup_data.py` — migration utility

If images were accidentally placed in `data/images/train|val/` instead of
`dataset/<ClassName>/`, this script moves them back.

```bash
python scripts/setup_data.py            # dry-run (shows what would move)
python scripts/setup_data.py --execute  # actually moves the files
```

---

## Training Reference

```bash
# Minimal — CPU, good first test
python !training/train.py --epochs 30 --imgsz 640 --device cpu

# GPU
python !training/train.py --epochs 50 --imgsz 640 --batch 16 --device 0

# All options
python !training/train.py \
    --model   weights/yolov8n.pt \
    --data    !training/data.yaml \
    --epochs  50 \
    --imgsz   640 \
    --batch   16 \
    --device  cpu \
    --name    pcb_solder_v1 \
    --patience 20
```

| Tip | Detail |
|---|---|
| Image size | 640 is standard; use 320 to train faster on CPU |
| Batch size | Reduce to 8 or 4 if RAM is limited |
| Larger models | `--model yolov8s.pt` or `yolov8m.pt` for more accuracy |
| Early stopping | `--patience 20` — stops if val loss plateaus for 20 epochs |
| Outputs | `weights/best.pt`, `weights/last.pt`, `runs/detect/<name>/` |

---

## Cloud Run Deployment

```bash
REGION=us-central1
PROJECT=$(gcloud config get-value project)
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/chipset-defect-vision/app:latest"

gcloud artifacts repositories create chipset-defect-vision \
    --repository-format=docker --location=${REGION}
gcloud auth configure-docker ${REGION}-docker.pkg.dev

docker build -t ${IMAGE} .
docker push ${IMAGE}

gcloud run deploy chipset-defect-vision \
    --image ${IMAGE} --region ${REGION} \
    --platform managed --allow-unauthenticated \
    --memory 2Gi --cpu 2 --concurrency 10 --timeout 60 --port 8080
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `segment_anything not installed` | Activate `.venv-sam` → `pip install -r requirements-sam.txt` |
| `[SAM] ERROR: Checkpoint not found` | Download a SAM checkpoint (see Step 0b above) and copy to `weights/` |
| `[SAM] ERROR: No recognised class folders` | Ensure `dataset/` contains folders named exactly as the 6 classes |
| `[SAM] WARNING: Cannot infer model type` | Rename checkpoint to include `vit_b`, `vit_l`, or `vit_h` |
| `[Annotator] ERROR: No display detected` | Run `annotate.py` on a local workstation (not headless Docker) |
| `[YOLO-Label] ERROR: Weights not found` | Run `!training/train.py` first to produce `weights/best.pt` |
| `[Predictor] ERROR: No weight files found` | Copy `yolov8n.pt` to `weights/` or rebuild the Docker image |
| `[Train] ERROR: Model weights not found` | Ensure `weights/yolov8n.pt` exists; run from project root |
| `[Train] ERROR: Dataset is not ready` | Run `scripts/sam_auto_annotate.py` first |
| `status: DEFECT` on clean boards | Lower `CONF_THRESHOLD` in `predictor.py` (default 0.30) |
| `status: GOOD` on defective boards | Run more bootstrap iterations; increase training epochs |
| Camera black screen | Serve over HTTPS or use `localhost` |
| Docker build OOM | Add `--memory 6g` to `docker build` |
| `CopyIgnoredFile` build error | Ensure `training/` is in `.dockerignore` and removed from Dockerfile `COPY` |

---

## License

MIT
