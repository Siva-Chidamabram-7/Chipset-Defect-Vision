# Chipset Defect Vision

**YOLOv8-powered PCB solder defect detection** — FastAPI backend · Glassmorphic UI · Docker · Fully offline

---

## Features

| | |
|---|---|
| Upload or webcam capture | Drag-and-drop or live camera frame |
| YOLOv8 inference | Bounding boxes, confidence scores, class labels |
| Two classes | `good` · `defect` |
| Annotated result | Base64-encoded JPEG returned with JSON |
| Glassmorphic UI | Frosted-glass panels, smooth animations, zero frameworks |
| Offline-first | Zero network calls at runtime; all assets baked in |
| Docker ready | Single-stage build, CPU-only torch, non-root runtime user |

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
├── data/                     ← dataset (images + labels, git-ignored)
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
├── raw_data/                 ← pre-split source images + LabelImg annotations
│   ├── images/
│   └── labels/
├── training/
│   ├── data.yaml             ← YOLO dataset config (canonical)
│   ├── train.py              ← fine-tuning script (offline pre-flight checks)
│   └── prepare_dataset.py    ← train/val splitter
├── weights/                  ← model weights (git-ignored; baked in Docker)
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 1 · Offline Deployment

> **This system is designed for air-gapped environments** (factory floors,
> edge servers, or any machine without internet access).

### How offline operation works

| Stage | Has internet? | What happens |
|-------|--------------|--------------|
| `docker build` | ✅ Yes (build machine) | Downloads Python deps + YOLOv8n weights; bakes everything into the image |
| `docker run` | ❌ Not required | Loads all weights from disk; zero outbound network calls |

### Model loading order (predictor.py)

The server checks for weights in this exact order:

```
1. weights/best.pt    ← your fine-tuned model (mount via -v or copy before build)
2. weights/yolov8n.pt ← base YOLOv8n baked into the Docker image at build time
```

If **neither file exists** the server still starts and returns a visible
`"NO MODEL LOADED"` overlay on every prediction — it never attempts a download.

### Build once, run anywhere

```bash
# On a machine WITH internet:
docker build -t chipset-defect-vision .

# Save the image for transfer to an offline machine:
docker save chipset-defect-vision | gzip > chipset-defect-vision.tar.gz

# On the offline machine:
docker load < chipset-defect-vision.tar.gz
docker run --network none -p 8080:8080 chipset-defect-vision
```

> `--network none` is optional but explicitly proves no network is needed.

### Injecting a fine-tuned model without rebuilding

```bash
docker run --network none -p 8080:8080 \
    -v "$(pwd)/weights/best.pt:/app/weights/best.pt:ro" \
    chipset-defect-vision
```

---

## 2 · Local Development Setup

### Prerequisites

- Python 3.10+
- pip

### Install dependencies

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# CPU-only torch (avoids pulling the 2 GB GPU bundle)
pip install torch==2.3.0 torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

> **Offline local dev:** download the torch wheel on a connected machine and
> install from the local `.whl` file:
> `pip install torch-2.3.0+cpu-cp310-*.whl --no-index`

### Run the server

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

Open http://localhost:8080

> **First run — no best.pt?**  Copy `weights/yolov8n.pt` into the `weights/`
> folder (download it once on a connected machine).  The server will use it
> automatically.  It will **not** download anything at runtime.

---

## 3 · Dataset Preparation (Open-Source, Offline Workflow)

All labeling is done with **LabelImg** — a free, offline desktop tool.
No cloud upload, no API keys, no Roboflow account required.

### 3.1 · Collect images

Photograph PCB boards under consistent, diffuse lighting:
- Resolution: 640 × 640 px or higher
- Coverage: capture every solder joint angle
- Balance: aim for ≥ 50 images per class

Place raw images in:
```
raw_data/images/
```

### 3.2 · Install LabelImg

```bash
pip install labelImg
```

Or as a standalone binary: https://github.com/HumanSignal/labelImg/releases

### 3.3 · Configure LabelImg for YOLO format

1. Launch:
   ```bash
   labelImg
   ```
2. **Open Dir** → select `raw_data/images/`
3. **Change Save Dir** → select `raw_data/labels/`
4. Click **PascalVOC** (bottom-left toggle) until it shows **YOLO**
5. Create `raw_data/labels/classes.txt` with exactly:
   ```
   good
   defect
   ```
   LabelImg reads this file for class names.

### 3.4 · Draw bounding boxes

| Class | What to box | Color in UI |
|-------|------------|-------------|
| `good` | Clean, shiny, properly-formed solder joint | Green |
| `defect` | Cold joint, bridge, void, lifted pad, insufficient solder | Red |

**Labeling rules:**
- Box tightly around the **solder joint only** — not the entire component or board
- One box per joint — do not merge multiple joints into one box
- Skip joints that are partially out of frame
- If a joint is ambiguous, skip it rather than guessing

**Keyboard shortcuts:**
| Key | Action |
|-----|--------|
| `W` | Draw bounding box |
| `D` | Next image |
| `A` | Previous image |
| `Ctrl+S` | Save current labels |
| `Del` | Delete selected box |

### 3.5 · YOLO label format

Each image gets a `.txt` file in `raw_data/labels/` with the same stem:

```
raw_data/labels/board_001.txt
```

Each line = one object:
```
<class_id> <x_center> <y_center> <width> <height>
```

All values are **normalised 0–1** relative to image dimensions.

**Example** — two joints on a 640×640 image:
```
0 0.421875 0.328125 0.125000 0.093750
1 0.671875 0.500000 0.156250 0.109375
```
- Line 1: class `0` (good),   centre (0.42, 0.33), size 80×60 px
- Line 2: class `1` (defect), centre (0.67, 0.50), size 100×70 px

### 3.6 · Split dataset into train / val

```bash
python training/prepare_dataset.py
```

Default split: 80 % train · 20 % val.

Custom split:
```bash
python training/prepare_dataset.py \
    --src raw_data \
    --dst data \
    --train 0.80 \
    --val  0.20
```

Output:
```
data/
├── images/
│   ├── train/   (e.g. 80 images)
│   └── val/     (e.g. 20 images)
└── labels/
    ├── train/
    └── val/
```

---

## 4 · Training

> **Offline requirement:** `weights/yolov8n.pt` must exist locally.
> Download it once on a connected machine, then transfer it to `weights/`.

```bash
# Minimal (CPU, 30 epochs — good for a first test run)
python training/train.py --epochs 30 --imgsz 640 --device cpu

# Recommended (GPU — replace '0' with your GPU index)
python training/train.py --epochs 50 --imgsz 640 --batch 16 --device 0

# Explicit paths (if weights are in a custom location)
python training/train.py \
    --model weights/yolov8n.pt \
    --data  training/data.yaml \
    --epochs 50
```

`train.py` performs pre-flight checks before starting:
- Verifies `weights/yolov8n.pt` (or your `--model` path) exists locally
- Verifies `data/images/train/` and `data/images/val/` contain images
- Fails with a clear error message if anything is missing — **no silent downloads**

Best weights are automatically copied to `weights/best.pt` after training.

### Training tips

| Tip | Detail |
|-----|--------|
| Image size | 640 is the sweet spot; use 320 to train faster on CPU |
| Batch size | Reduce to 8 or 4 if you run out of RAM |
| Augmentation | Mosaic + MixUp enabled by default in train.py |
| Early stopping | `--patience 20` stops if val loss doesn't improve |
| Larger models | `--model yolov8s.pt` or `yolov8m.pt` for better accuracy |

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
    {
      "label":      "defect",
      "confidence": 0.914,
      "bbox":       [120, 45, 210, 130]
    }
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
> Must be served over **HTTPS** or `localhost` in Chrome/Edge.

---

## 7 · Docker

### Build (requires internet — one-time only)

```bash
docker build -t chipset-defect-vision .
```

The build:
1. Installs all Python dependencies (torch CPU, ultralytics, FastAPI, etc.)
2. Downloads `yolov8n.pt` into `weights/` inside the image
3. Verifies the weight file exists — **fails the build** if it doesn't
4. Copies all application code and frontend

### Run (fully offline)

```bash
docker run -p 8080:8080 chipset-defect-vision

# Explicitly disable networking to prove no internet is needed:
docker run --network none -p 8080:8080 chipset-defect-vision
```

### Mount fine-tuned weights (no rebuild required)

```bash
docker run --network none -p 8080:8080 \
    -v "$(pwd)/weights/best.pt:/app/weights/best.pt:ro" \
    chipset-defect-vision
```

### Transfer to an offline machine

```bash
# On connected machine — save image to a tar archive
docker save chipset-defect-vision | gzip > chipset-defect-vision.tar.gz

# Copy the archive to the offline machine (USB / secure file transfer)
# On offline machine — load and run
docker load < chipset-defect-vision.tar.gz
docker run --network none -p 8080:8080 chipset-defect-vision
```

### GPU support (NVIDIA)

Change the torch install line in Dockerfile Layer 2:
```dockerfile
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
Run with `--gpus all`.

---

## 8 · Google Cloud Run Deployment

> For cloud deployments the same offline image works — the build still needs
> internet, but the running container doesn't.

### Prerequisites

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com artifactregistry.googleapis.com
```

### Build & push

```bash
REGION=us-central1
PROJECT=$(gcloud config get-value project)
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/chipset-defect-vision/app:latest"

gcloud artifacts repositories create chipset-defect-vision \
    --repository-format=docker --location=${REGION}

gcloud auth configure-docker ${REGION}-docker.pkg.dev

docker build -t ${IMAGE} .
docker push ${IMAGE}
```

### Deploy

```bash
gcloud run deploy chipset-defect-vision \
    --image ${IMAGE} \
    --region ${REGION} \
    --platform managed \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --concurrency 10 \
    --timeout 60 \
    --port 8080
```

---

## 9 · Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8080` | Listening port (set automatically by Cloud Run) |
| `YOLO_CONFIG_DIR` | `/app/.yolo` | Ultralytics config write location (keep out of /root) |
| `YOLO_TELEMETRY` | `False` | Disables outbound analytics |
| `YOLO_AUTOINSTALL` | `False` | Prevents ultralytics from auto-installing packages |
| `HF_DATASETS_OFFLINE` | `1` | HuggingFace offline mode |
| `TRANSFORMERS_OFFLINE` | `1` | Transformers offline mode |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `[Predictor] ERROR: No weight files found` | Copy `yolov8n.pt` into `weights/`, or rebuild the Docker image |
| `[Train] ERROR: Model weights not found` | Run `python training/train.py` from the project root; ensure `weights/yolov8n.pt` exists |
| `[Prepare] ERROR: images directory not found` | Create `raw_data/images/` and place your images there |
| Camera black screen | Serve over HTTPS or use `localhost` |
| Camera permission denied | Allow camera in browser site settings |
| `0 detections` on inference | Use fine-tuned `weights/best.pt`; base model only knows COCO classes |
| Docker build OOM | Add `--memory 6g` to `docker build` |
| Docker build — weight download fails | Check build-machine internet; proxy settings may block PyPI / GitHub |

---

## License

MIT
