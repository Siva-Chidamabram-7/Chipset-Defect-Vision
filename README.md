# Chipset Defect Vision

**Hybrid SAM + YOLOv8 PCB solder defect detection** — FastAPI · Glassmorphic UI · Docker · Fully offline

---

## Features

| | |
|---|---|
| Hybrid pipeline | SAM auto-proposes regions → human labels → YOLO trains |
| Upload or webcam capture | Drag-and-drop or live camera frame |
| YOLOv8 inference | Bounding boxes, confidence scores, class labels |
| Two classes | `good` · `defect` |
| Annotated result | Base64-encoded JPEG returned with JSON |
| Glassmorphic UI | Frosted-glass panels, smooth animations, zero frameworks |
| Offline-first | Zero network calls at runtime; all assets baked in |
| Docker ready | Single-stage build, CPU-only torch, non-root runtime user |

---

## Hybrid Pipeline Overview

```
Phase 1 — Region Detection (SAM)
─────────────────────────────────
raw_data/images/*.jpg
    │
    ▼  python scripts/generate_regions.py
raw_data/regions/*.json          ← bounding box proposals per image


Phase 2 — Human Labeling (annotate.py)
───────────────────────────────────────
raw_data/regions/*.json
    │
    ▼  python scripts/annotate.py
raw_data/labels/*.txt            ← YOLO-format labels  (class bbox …)


Phase 3 — Dataset Split (prepare_dataset.py)
─────────────────────────────────────────────
raw_data/images/ + raw_data/labels/
    │
    ▼  python training/prepare_dataset.py
data/
  images/train/ + images/val/
  labels/train/ + labels/val/


Phase 4 — YOLO Fine-tuning (train.py)
──────────────────────────────────────
data/ + weights/yolov8n.pt
    │
    ▼  python training/train.py
weights/best.pt                  ← your fine-tuned PCB model


Phase 5 — Inference (FastAPI + YOLO only)
──────────────────────────────────────────
weights/best.pt
    │
    ▼  docker run / uvicorn
POST /predict  →  JSON detections + annotated image
```

> **SAM is used only during dataset creation (Phases 1-2).**
> **YOLO is used only during inference (Phase 5).**
> They are never loaded simultaneously.

---

## Architecture — Why SAM + YOLO?

| Concern | SAM | YOLO |
|---------|-----|------|
| Purpose | Region detection — finds *where* solder joints are | Object detection — classifies *what* is in a region |
| When used | Dataset creation (once) | Inference (every prediction) |
| Requires display | Yes (annotation tool) | No |
| Inference speed | ~2–5 s / image on CPU | ~50–200 ms / image on CPU |
| In Docker image | No (dataset creation happens before Docker) | Yes |

SAM dramatically reduces labeling time: instead of manually drawing hundreds of boxes, a human simply presses **G** / **D** on each SAM-proposed region — usually 10× faster.

---

## Project Structure

```
Chipset-Defect-Vision/
├── app/
│   ├── main.py               ← FastAPI app — /predict, /health
│   ├── model/
│   │   └── predictor.py      ← YOLOv8 inference + annotation (offline-only)
│   └── utils/
│       └── image_utils.py    ← base64 decode, image validation
├── data/                     ← split dataset (git-ignored content)
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── frontend/
│   ├── index.html
│   ├── styles.css            ← glassmorphic design system
│   └── script.js             ← vanilla JS, async/await
├── raw_data/                 ← pre-split source (git-ignored content)
│   ├── images/               ← original PCB photos
│   ├── labels/               ← YOLO .txt (output of annotate.py)
│   └── regions/              ← SAM JSON proposals (output of generate_regions.py)
├── scripts/
│   ├── generate_regions.py   ← Phase 1: SAM → JSON region proposals
│   └── annotate.py           ← Phase 2: OpenCV interactive labeling tool
├── training/
│   ├── data.yaml             ← YOLO dataset config (canonical)
│   ├── train.py              ← YOLO fine-tuning (offline, pre-flight checked)
│   └── prepare_dataset.py    ← train/val splitter
├── weights/                  ← model weights (git-ignored; baked in Docker)
│   ├── yolov8n.pt            ← YOLO base  (baked in Docker at build time)
│   ├── sam_vit_b.pth         ← SAM base   (download once, transfer offline)
│   └── best.pt               ← fine-tuned (produced by train.py)
├── Dockerfile
├── requirements.txt          ← inference dependencies
├── requirements-sam.txt      ← dataset-creation dependencies (SAM + OpenCV GUI)
└── README.md
```

---

## 1 · Offline Deployment

> **This system is designed for air-gapped environments** (factory floors,
> edge servers, or any machine without internet access).

### How offline operation works

| Stage | Internet? | What happens |
|-------|-----------|--------------|
| SAM checkpoint download | ✅ Once | `wget` on connected machine → transfer to `weights/` |
| `pip install -r requirements-sam.txt` | ✅ Once | Install on dataset-creation workstation |
| `docker build` | ✅ Once | Downloads Python deps + YOLOv8n weights |
| Dataset creation + training | ❌ None | All local — SAM, annotate, train scripts |
| `docker run` (inference) | ❌ None | Loads weights from disk; zero outbound calls |

### Model loading order (predictor.py)

```
1. weights/best.pt    ← fine-tuned model (mount via -v or copy before build)
2. weights/yolov8n.pt ← base YOLOv8n baked into the Docker image at build time
```

If neither file exists the server still starts and returns a visible
`"NO MODEL LOADED"` banner — it never attempts a download.

### Build once, run anywhere

```bash
# On a machine WITH internet:
docker build -t chipset-defect-vision .

# Save for transfer to an offline machine:
docker save chipset-defect-vision | gzip > chipset-defect-vision.tar.gz

# On the offline machine:
docker load < chipset-defect-vision.tar.gz
docker run --network none -p 8080:8080 chipset-defect-vision
```

---

## 2 · Local Development Setup

### Prerequisites

- Python 3.10+
- A graphical display (for `scripts/annotate.py`)

### Install inference dependencies

```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
source .venv/bin/activate  # macOS / Linux

pip install torch==2.3.0 torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Install dataset-creation dependencies (separate venv recommended)

```bash
python -m venv .venv-sam
.venv-sam\Scripts\activate     # Windows
source .venv-sam/bin/activate  # macOS / Linux

pip install torch==2.3.0 torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-sam.txt
```

### Run the inference server locally

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

Open http://localhost:8080

---

## 3 · Dataset Preparation — Hybrid SAM + LabelImg Workflow

### 3.1 · Collect images

Place raw PCB photos in:
```
raw_data/images/
```

Photography tips:
- Consistent overhead lighting (no harsh shadows)
- Resolution ≥ 640 × 640 px
- One board per image or cropped to the region of interest

---

### 3.2 · Download SAM checkpoint (once, on a connected machine)

```bash
# ViT-B — recommended (375 MB, good speed/accuracy balance on CPU)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
     -O weights/sam_vit_b.pth

# ViT-L — better accuracy (1.2 GB, slower on CPU)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth \
     -O weights/sam_vit_l.pth

# ViT-H — best accuracy (2.4 GB, needs GPU for reasonable speed)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth \
     -O weights/sam_vit_h.pth
```

Transfer `weights/sam_vit_b.pth` to your offline workstation.

---

### 3.3 · Phase 1 — SAM region detection

```bash
# Default (uses raw_data/images → raw_data/regions, vit_b, CPU)
python scripts/generate_regions.py

# Explicit options
python scripts/generate_regions.py \
    --images      raw_data/images \
    --output      raw_data/regions \
    --checkpoint  weights/sam_vit_b.pth \
    --model-type  vit_b \
    --device      cpu
```

**Tuning the region filter** (adjust if you get too many or too few proposals):

| Flag | Default | When to change |
|------|---------|----------------|
| `--min-area` | 400 | Increase if small dust/noise is proposed |
| `--max-area` | 40000 | Decrease if large background patches appear |
| `--min-aspect` | 0.25 | Decrease if tall narrow joints are missed |
| `--max-aspect` | 4.0 | Decrease to exclude wires/traces |
| `--points-per-side` | 32 | Increase to 64 for denser, slower proposal |
| `--nms-thresh` | 0.5 | Decrease to keep fewer overlapping boxes |

Output: one JSON file per image in `raw_data/regions/`:
```json
{
  "image": "raw_data/images/board_001.jpg",
  "width": 640, "height": 480,
  "regions": [
    {
      "id": 0,
      "bbox_abs":  [x1, y1, x2, y2],
      "bbox_yolo": [cx, cy, w, h],
      "area": 1234,
      "stability_score": 0.92,
      "label": null
    }
  ]
}
```

---

### 3.4 · Phase 2 — Human labeling (annotate.py)

> **Requires a graphical display** (not headless).

```bash
python scripts/annotate.py

# Custom paths
python scripts/annotate.py \
    --regions raw_data/regions \
    --images  raw_data/images \
    --output  raw_data/labels \
    --zoom    3
```

**Two OpenCV windows open:**

| Window | Shows |
|--------|-------|
| `PCB Annotator` | Full PCB image; previously labeled boxes in green/red; current box highlighted in cyan |
| `Current Region` | Zoomed crop of the current region + status info |

**Keyboard controls:**

| Key | Action |
|-----|--------|
| `G` | Label current region as **good** (class 0) |
| `D` | Label current region as **defect** (class 1) |
| `S` | **Skip** this region (not written to label file) |
| `B` | **Undo** last label in the current image |
| `Q` or `Esc` | Save current image labels and quit |

**Labeling rules:**
- Label tight around the solder joint — SAM proposals are already cropped
- If a proposal covers an entire component body (not a joint), press `S` to skip
- If a proposal is two joints merged together, press `S` — better to skip than mis-label
- Aim for ≥ 50 examples per class for meaningful training

**Resume:** if you quit mid-session, just run `annotate.py` again.
Images with existing `.txt` label files are skipped automatically.

Output: YOLO `.txt` files in `raw_data/labels/`:
```
0 0.421875 0.328125 0.125000 0.093750
1 0.671875 0.500000 0.156250 0.109375
```

---

### 3.5 · Alternative: LabelImg (manual bounding boxes)

If you prefer to draw boxes manually instead of using SAM proposals:

```bash
pip install labelImg
labelImg raw_data/images raw_data/labels
```

1. Switch format to **YOLO** (bottom-left toggle)
2. Create `raw_data/labels/classes.txt`:
   ```
   good
   defect
   ```
3. Draw boxes → `Ctrl+S` → next image

Both workflows produce the same YOLO `.txt` format — the two approaches are interchangeable.

---

### 3.6 · Phase 3 — Train / val split

```bash
python training/prepare_dataset.py
```

Default split: **80 % train · 20 % val**.

Custom:
```bash
python training/prepare_dataset.py \
    --src raw_data \
    --dst data \
    --train 0.80 --val 0.20
```

Output:
```
data/
├── images/train/   (80 % of images)
├── images/val/     (20 % of images)
├── labels/train/
└── labels/val/
```

---

## 4 · Training

> **Offline requirement:** `weights/yolov8n.pt` must exist locally.

```bash
# Minimal — CPU, 30 epochs (good first test)
python training/train.py --epochs 30 --imgsz 640 --device cpu

# Recommended — GPU (replace '0' with your GPU index)
python training/train.py --epochs 50 --imgsz 640 --batch 16 --device 0

# Explicit paths
python training/train.py \
    --model  weights/yolov8n.pt \
    --data   training/data.yaml \
    --epochs 50
```

Pre-flight checks before training starts:
- Verifies base model weights exist locally — **no silent downloads**
- Verifies train/val image directories are non-empty
- Exits with actionable error messages if anything is missing

Best weights are copied to `weights/best.pt` automatically.

### Training tips

| Tip | Detail |
|-----|--------|
| Image size | 640 is the sweet spot; use 320 to train faster on CPU |
| Batch size | Reduce to 8 or 4 if RAM is limited |
| Augmentation | Mosaic + MixUp enabled by default |
| Early stopping | `--patience 20` — stops if val loss plateaus |
| Larger models | `--model yolov8s.pt` or `yolov8m.pt` for more accuracy |

---

## 5 · API Usage

### Health check

```bash
curl http://localhost:8080/health
```
```json
{ "status": "ok", "model_loaded": true, "version": "1.0.0" }
```

### Predict from file upload

```bash
curl -X POST http://localhost:8080/predict \
     -F "file=@pcb_sample.jpg"
```

### Predict from base64

```bash
B64=$(base64 -i pcb_sample.jpg)
curl -X POST http://localhost:8080/predict \
     -F "image_base64=${B64}"
```

### Response schema

```json
{
  "detections": [
    { "label": "defect", "confidence": 0.914, "bbox": [120, 45, 210, 130] }
  ],
  "total":        3,
  "defect_count": 1,
  "good_count":   2,
  "image":        "<base64-encoded-annotated-jpeg>",
  "model":        "fine-tuned"
}
```

---

## 6 · Camera Usage

1. Click **"Live Camera"** tab
2. Click **"Start Camera"** → allow browser permission
3. Point camera at PCB
4. Click **"Capture Frame"**
5. Click **"Scan Image"**

> Camera uses `navigator.mediaDevices.getUserMedia`.
> Must be served over **HTTPS** or `localhost`.

---

## 7 · Docker

### Build (requires internet — one-time only)

```bash
docker build -t chipset-defect-vision .
```

The build:
1. Installs Python dependencies (torch CPU, ultralytics, FastAPI …)
2. Downloads `yolov8n.pt` into `/app/weights/`
3. **Fails the build** if the weight file is missing after download
4. Copies all app code, frontend, scripts, and training files

### Run (fully offline)

```bash
docker run -p 8080:8080 chipset-defect-vision

# Prove no internet is needed:
docker run --network none -p 8080:8080 chipset-defect-vision
```

### Mount fine-tuned weights

```bash
docker run --network none -p 8080:8080 \
    -v "$(pwd)/weights/best.pt:/app/weights/best.pt:ro" \
    chipset-defect-vision
```

### Transfer to an offline machine

```bash
# Connected machine
docker save chipset-defect-vision | gzip > chipset-defect-vision.tar.gz

# Offline machine
docker load < chipset-defect-vision.tar.gz
docker run --network none -p 8080:8080 chipset-defect-vision
```

### Enable SAM inside the container (optional)

Uncomment the SAM layer in the Dockerfile and mount the checkpoint:

```dockerfile
# In Dockerfile, uncomment:
RUN pip install --no-cache-dir segment-anything==1.0
```

```bash
docker build -t chipset-defect-vision .
docker run --network none -p 8080:8080 \
    -v "$(pwd)/weights/sam_vit_b.pth:/app/weights/sam_vit_b.pth:ro" \
    -v "$(pwd)/raw_data:/app/raw_data" \
    chipset-defect-vision \
    python scripts/generate_regions.py
```

---

## 8 · Google Cloud Run Deployment

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

## 9 · Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8080` | Listening port |
| `YOLO_CONFIG_DIR` | `/app/.yolo` | Ultralytics config write location |
| `YOLO_TELEMETRY` | `False` | Disables outbound analytics |
| `YOLO_AUTOINSTALL` | `False` | Prevents auto-package-install |
| `HF_DATASETS_OFFLINE` | `1` | HuggingFace offline mode |
| `TRANSFORMERS_OFFLINE` | `1` | Transformers offline mode |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `segment-anything not installed` | Activate `.venv-sam` and run `pip install -r requirements-sam.txt` |
| `[SAM] ERROR: Checkpoint not found` | Download `sam_vit_b.pth` and copy to `weights/` |
| `[Annotator] ERROR: No display detected` | Run `annotate.py` on a local workstation (not in headless Docker) |
| `[SAM] 0 regions after filtering` | Tune `--min-area`, `--max-area`, `--points-per-side`; try `--overwrite` |
| `[Predictor] ERROR: No weight files found` | Copy `yolov8n.pt` to `weights/` or rebuild Docker image |
| `[Train] ERROR: Model weights not found` | Ensure `weights/yolov8n.pt` exists; run from project root |
| Camera black screen | Serve over HTTPS or use `localhost` |
| `0 detections` on inference | Use fine-tuned `weights/best.pt`; base COCO model doesn't know solder classes |
| Docker build OOM | Add `--memory 6g` to `docker build` |

---

## License

MIT
