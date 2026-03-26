# Chipset Defect Vision

Chipset Defect Vision is a YOLOv8-based defect detection system with a strict split between:

- `app/` for offline inference only
- `training/` for local and Vertex AI GPU training only

The inference service is designed for Cloud Run and will only start when `weights/best.pt` is present. It does not download models at runtime and does not fall back to base YOLO checkpoints.

## Production Guarantees

- Inference only uses `weights/best.pt`
- Missing model files fail fast at startup and at Docker build time
- The inference Docker image contains only `app/`, `frontend/`, `requirements.txt`, and `weights/best.pt`
- Training supports local datasets or GCS datasets staged locally before YOLO runs
- Vertex AI training is configured for project `detection-490708`, region `asia-south1`, and `NVIDIA_TESLA_T4` by default

## Folder Structure

```text
Chipset-Defect-Vision/
├── app/
│   ├── config.py
│   ├── main.py
│   ├── model/
│   │   └── predictor.py
│   ├── schemas.py
│   └── utils/
│       └── image_utils.py
├── frontend/
├── scripts/
│   └── test_inference.py
├── training/
│   ├── config.py
│   ├── data.yaml
│   ├── Dockerfile
│   ├── Dockerfile.dockerignore
│   ├── requirements-training.txt
│   ├── train.py
│   ├── vertex_job.py
│   └── scripts/
│       ├── auto_label_with_yolo.py
│       ├── check_pipeline.py
│       ├── gcs_utils.py
│       ├── sam_auto_annotate.py
│       ├── sam_to_yolo.py
│       └── setup_data.py
├── data/
├── raw_data/
├── weights/
│   └── best.pt
├── Dockerfile
└── requirements.txt
```

## Local Inference Setup

Install dependencies:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.3.0 torchvision==0.18.0
pip install -r requirements.txt
```

Start the API:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Health check:

```bash
curl http://127.0.0.1:8080/health
```

Predict with a file:

```bash
curl -X POST http://127.0.0.1:8080/predict \
  -F "file=@sample.jpg"
```

Predict with base64:

```bash
curl -X POST http://127.0.0.1:8080/predict \
  -F "image_base64=$(base64 -w 0 sample.jpg)"
```

Run the latency hook:

```bash
python scripts/test_inference.py data/images/val/example.jpg
```

## Cloud Run Inference

Build the inference image:

```bash
docker build -t chipset-defect-vision-inference .
```

The build fails if `weights/best.pt` is missing or suspiciously small.

Run locally:

```bash
docker run --rm -p 8080:8080 chipset-defect-vision-inference
```

Deploy to Cloud Run:

```bash
gcloud run deploy chipset-defect-vision \
  --image asia-south1-docker.pkg.dev/detection-490708/chipset-defect-vision/inference:latest \
  --region asia-south1 \
  --platform managed \
  --port 8080 \
  --allow-unauthenticated
```

## Local Training

Install the training environment in a separate virtual environment:

```bash
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r training/requirements-training.txt
```

Train against the local dataset:

```bash
python training/train.py \
  --data training/data.yaml \
  --model weights/yolov8n.pt \
  --epochs 50 \
  --batch 16 \
  --lr 0.01 \
  --imgsz 640 \
  --optimizer AdamW \
  --device cuda
```

The best checkpoint is copied to `weights/best.pt` after each run.

## Vertex AI Training

Build and push the training image:

```bash
docker build -f training/Dockerfile -t asia-south1-docker.pkg.dev/detection-490708/chipset-defect-vision/training:latest .
docker push asia-south1-docker.pkg.dev/detection-490708/chipset-defect-vision/training:latest
```

Submit a Vertex AI job with the Python launcher:

```bash
python training/vertex_job.py \
  --image-uri asia-south1-docker.pkg.dev/detection-490708/chipset-defect-vision/training:latest \
  --data gs://chip-defect-vision-bucket/data/ \
  --model gs://chip-defect-vision-bucket/models/yolov8n.pt \
  --output-model gs://chip-defect-vision-bucket/data/ \
  --epochs 50 \
  --batch 16 \
  --lr 0.01 \
  --imgsz 640 \
  --optimizer AdamW \
  --device cuda
```

Equivalent direct training command inside the container:

```bash
python training/train.py \
  --data gs://chip-defect-vision-bucket/data/ \
  --model gs://chip-defect-vision-bucket/models/yolov8n.pt \
  --output-model gs://chip-defect-vision-bucket/data/ \
  --epochs 50 \
  --batch 16 \
  --lr 0.01 \
  --imgsz 640 \
  --optimizer AdamW \
  --device cuda
```

## Hyperparameters

`training/train.py` supports both CLI flags and environment variables.

| Setting | CLI flag | Environment variable | Default |
|---|---|---|---|
| Base model | `--model` | `YOLO_BASE_MODEL` | `weights/yolov8n.pt` |
| Dataset | `--data` | `TRAIN_DATA_PATH`, `AIP_TRAINING_DATA_URI` | `training/data.yaml` |
| Output model | `--output-model` | `TRAIN_OUTPUT_MODEL`, `AIP_MODEL_DIR` | disabled |
| Epochs | `--epochs` | `TRAIN_EPOCHS` | `50` |
| Batch size | `--batch` | `TRAIN_BATCH_SIZE` | `16` |
| Learning rate | `--lr` | `TRAIN_LEARNING_RATE` | `0.001` |
| Image size | `--imgsz` | `TRAIN_IMAGE_SIZE` | `640` |
| Optimizer | `--optimizer` | `TRAIN_OPTIMIZER` | `AdamW` |
| Device | `--device` | `TRAIN_DEVICE` | `cpu` |
| Run name | `--name` | `TRAIN_RUN_NAME` | `pcb_solder_vertex` |
| Patience | `--patience` | `TRAIN_PATIENCE` | `20` |

## GCS Usage

Training accepts either:

- a local dataset YAML such as `training/data.yaml`
- a local dataset directory such as `data/`
- a GCS prefix such as `gs://chip-defect-vision-bucket/data/`

When a GCS dataset is provided:

1. the dataset is downloaded to a temporary local staging directory
2. `data.yaml` is reused if present, otherwise one is generated automatically
3. YOLO training runs against the staged local copy
4. `best.pt` is uploaded to `--output-model` if that destination is a `gs://` path

Example:

```bash
python training/train.py \
  --data gs://chip-defect-vision-bucket/data/ \
  --model gs://chip-defect-vision-bucket/models/yolov8n.pt \
  --output-model gs://chip-defect-vision-bucket/data/ \
  --epochs 50 \
  --batch 16 \
  --lr 0.01 \
  --device cuda
```

## Training Helper Scripts

These utilities stay under `training/scripts/` and are intentionally excluded from the inference image:

- `sam_auto_annotate.py` generates YOLO labels from class-folder datasets
- `auto_label_with_yolo.py` rewrites labels using the latest trained model
- `check_pipeline.py` checks training assets and checkpoints
- `setup_data.py` helps migrate misplaced local images

## Notes

- `weights/best.pt` is mandatory for inference
- `weights/yolov8n.pt` is optional and only used for training bootstrap
- No training code is copied into the Cloud Run inference image
- No model downloads occur at inference runtime
