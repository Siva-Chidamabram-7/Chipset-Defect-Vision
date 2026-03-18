"""
train.py — Fine-tune YOLOv8 on the PCB solder defect dataset
─────────────────────────────────────────────────────────────
Usage:
    python training/train.py [--epochs 50] [--imgsz 640] [--batch 16] [--device cpu]

Output:
    Best weights saved to:  weights/best.pt
    Last weights saved to:  weights/last.pt
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 PCB solder defect training")
    p.add_argument("--model",   default="yolov8n.pt",          help="Base model (yolov8n/s/m/l/x.pt)")
    p.add_argument("--data",    default="training/dataset.yaml",help="Path to dataset YAML")
    p.add_argument("--epochs",  type=int,   default=50)
    p.add_argument("--imgsz",   type=int,   default=640)
    p.add_argument("--batch",   type=int,   default=16)
    p.add_argument("--device",  default="0",                    help="'0','cpu','mps'")
    p.add_argument("--name",    default="pcb_solder_v1",        help="Run name")
    p.add_argument("--patience",type=int,   default=20,         help="Early-stop patience")
    return p.parse_args()


# ── Train ─────────────────────────────────────────────────────────────────────
def train(args):
    print(f"[Train] Loading base model: {args.model}")
    model = YOLO(args.model)

    print(f"[Train] Starting training → epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}")
    results = model.train(
        data      = args.data,
        epochs    = args.epochs,
        imgsz     = args.imgsz,
        batch     = args.batch,
        device    = args.device,
        name      = args.name,
        patience  = args.patience,
        # Augmentation
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        flipud=0.1,  fliplr=0.5,
        mosaic=1.0,  mixup=0.1,
        # Optimiser
        optimizer="AdamW",
        lr0=0.001, lrf=0.01,
        warmup_epochs=3,
        # Logging
        plots=True, save=True, save_period=10,
    )
    return results


# ── Copy best weights to /weights/ ────────────────────────────────────────────
def copy_weights(run_name: str):
    runs_dir = Path("runs/detect") / run_name / "weights"
    dest_dir = Path("weights")
    dest_dir.mkdir(exist_ok=True)

    for fname in ("best.pt", "last.pt"):
        src = runs_dir / fname
        if src.exists():
            dst = dest_dir / fname
            shutil.copy2(src, dst)
            print(f"[Train] Copied {src} → {dst}")
        else:
            print(f"[Train] Warning: {src} not found.")


# ── Validate ──────────────────────────────────────────────────────────────────
def validate(weights_path: str, data: str, imgsz: int, device: str):
    model = YOLO(weights_path)
    metrics = model.val(data=data, imgsz=imgsz, device=device)
    print(f"\n[Validate] mAP50:    {metrics.box.map50:.4f}")
    print(f"[Validate] mAP50-95: {metrics.box.map:.4f}")
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    train(args)
    copy_weights(args.name)
    best = f"weights/best.pt"
    if Path(best).exists():
        validate(best, args.data, args.imgsz, args.device)
    print("\n[Train] Done. Weights saved to weights/best.pt")
