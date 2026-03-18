# Chipset Defect Vision

**YOLOv8-powered PCB solder defect detection** — FastAPI backend · Glassmorphic UI · Docker · Google Cloud Run

---

## Features

| | |
|---|---|
| Upload or webcam capture | Drag-and-drop or live camera frame |
| YOLOv8 inference | Bounding boxes, confidence scores, class labels |
| Two classes | `good` · `defect` |
| Annotated result | Base64-encoded JPEG returned with JSON |
| Glassmorphic UI | Frosted-glass panels, smooth animations, zero frameworks |
| Docker ready | Multi-stage build, CPU + GPU compatible |
| Cloud Run ready | Stateless, port-agnostic via `$PORT` |

---

## Project Structure

```
Chipset-Defect-Vision/
├── app/
│   ├── main.py               ← FastAPI app, /predict, /health
│   ├── model/
│   │   └── predictor.py      ← YOLOv8 inference + annotation
│   └── utils/
│       └── image_utils.py    ← base64 decode, image validation
├── frontend/
│   ├── index.html
│   ├── styles.css            ← glassmorphic design system
│   └── script.js             ← vanilla JS, async/await
├── training/
│   ├── dataset.yaml          ← YOLO dataset config
│   ├── train.py              ← fine-tuning script
│   └── prepare_dataset.py    ← train/val/test splitter
├── weights/                  ← put best.pt here after training
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 1 · Local Setup

### Prerequisites
- Python 3.10+
- pip

### Install dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Run the server

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

Open http://localhost:8080

> **First run:** if `weights/best.pt` is missing, the system downloads `yolov8n.pt`
> automatically and uses it in demo mode.

---

## 2 · Dataset & Labeling

### Recommended datasets
- [PCBDATA (Kaggle)](https://www.kaggle.com/datasets/akhatova/pcb-defects) — 1386 images, 6 defect types
- [DeepPCB](https://github.com/tangsanli5201/DeepPCB) — aligned PCB pairs
- Custom capture: photograph boards under consistent lighting

### Labeling with Roboflow (recommended)
1. Upload images to [Roboflow](https://roboflow.com)
2. Draw bounding boxes → assign `good` or `defect`
3. Export as **YOLOv8** format
4. Download and place in `raw_data/`

### Labeling with LabelImg (local)
```bash
pip install labelImg
labelImg raw_data/images raw_data/labels
# Select YOLO format, draw boxes, save
```

### Class IDs
```
0 → good
1 → defect
```

### Prepare splits
```bash
python training/prepare_dataset.py \
    --src raw_data \
    --dst datasets/pcb_solder \
    --train 0.80 --val 0.15 --test 0.05
```

---

## 3 · Training

```bash
# Fine-tune YOLOv8 nano (fastest)
python training/train.py \
    --model yolov8n.pt \
    --epochs 50 \
    --imgsz 640 \
    --batch 16 \
    --device 0       # use 'cpu' if no GPU

# Larger models (better accuracy, slower)
# --model yolov8s.pt  (small)
# --model yolov8m.pt  (medium)
```

Best weights are automatically copied to `weights/best.pt`.

### Training tips
| Tip | Detail |
|-----|--------|
| Image size | 640 is the sweet spot; use 320 to train faster |
| Batch size | Reduce if OOM; increase for better GPU utilisation |
| Augmentation | Mosaic + MixUp enabled by default |
| Early stopping | `--patience 20` stops if no improvement |

---

## 4 · API Usage

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

## 5 · Camera Usage

1. Click **"Live Camera"** tab
2. Click **"Start Camera"** → allow browser permission
3. Point at PCB
4. Click **"Capture Frame"**
5. Click **"Scan Image"**

> Camera uses `navigator.mediaDevices.getUserMedia`.
> Must be served over **HTTPS** or `localhost` in Chrome.

---

## 6 · Docker

### Build
```bash
docker build -t chipset-defect-vision .
```

### Run
```bash
docker run -p 8080:8080 chipset-defect-vision
```

### With fine-tuned weights
```bash
docker run -p 8080:8080 \
    -v "$(pwd)/weights:/app/weights:ro" \
    chipset-defect-vision
```

### GPU support (NVIDIA)
Change the torch install line in Dockerfile:
```dockerfile
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
And run with `--gpus all`.

---

## 7 · Google Cloud Run Deployment

### Prerequisites
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com artifactregistry.googleapis.com
```

### Build & push to Artifact Registry
```bash
REGION=us-central1
PROJECT=$(gcloud config get-value project)
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/chipset-defect-vision/app:latest"

# Create repo (first time only)
gcloud artifacts repositories create chipset-defect-vision \
    --repository-format=docker --location=${REGION}

# Authenticate docker
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Build & push
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

### CI/CD with Cloud Build (optional)
```bash
gcloud builds submit --tag ${IMAGE}
```

---

## 8 · Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8080` | Listening port (set by Cloud Run automatically) |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: ultralytics` | `pip install ultralytics` |
| Camera black screen | Serve over HTTPS or localhost |
| Camera permission denied | Allow in browser site settings |
| Model returns 0 detections | Use fine-tuned `weights/best.pt`; base model only knows COCO classes |
| Docker build OOM | Add `--memory 4g` to `docker build` |

---

## License

MIT
