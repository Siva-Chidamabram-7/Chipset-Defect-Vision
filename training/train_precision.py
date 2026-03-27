"""
Phase 2 precision training — 5 classes, high-precision objective.
Run: python training/train_precision.py

WHY THIS FILE EXISTS:
  Previous training used AI-generated (Gemini) images and had only class 0 annotated.
  That caused precision ~0.2 (false positives everywhere).
  This script retrains from clean yolov8s.pt weights on real PCB images.

CONNECTED FILES:
  - training/data.yaml          : defines nc=5, class names, train/val paths
  - training/hyperparameters_precision.yaml : augmentation + loss weight reference doc
  - data/images/train/          : 310 real images (240 annotated + 70 background)
  - data/labels/train/          : 310 label files (70 are empty = background teaching)
  - runs/detect/pcb_precision_v1/weights/best.pt : output — used by mine_hard_negatives.py
  - training/scripts/mine_hard_negatives.py      : run AFTER this completes (Phase 3)
  - app/model/predictor.py      : loads weights/best.pt for inference (line 20)
"""
from pathlib import Path
from ultralytics import YOLO


def main():
    # Resolve project root from this file's location (training/ -> root)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # Points to training/data.yaml — defines nc=5 (5 active classes),
    # class names [Missing_hole, Mouse_bite, Open_circuit, Short, Spur],
    # and paths to data/images/train + data/images/val.
    # Classes 5,6,7 were DROPPED because they have 0 annotations.
    # See training/data.yaml line 6: nc: 5
    DATA_YAML = PROJECT_ROOT / "training" / "data.yaml"

    # Using yolov8s (small) not yolov8n (nano) — nano is too weak for small PCB defects.
    # NOT using best.pt — those weights were trained on fake Gemini images only for
    # class 0 (Missing_hole), making them poisoned for multi-class training.
    # yolov8s.pt gives clean ImageNet-pretrained backbone features.
    BASE_MODEL = PROJECT_ROOT / "yolov8s.pt"

    # Auto-download yolov8s if not already present in project root
    if not BASE_MODEL.exists():
        import urllib.request
        print("Downloading yolov8s.pt ...")
        urllib.request.urlretrieve(
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
            str(BASE_MODEL),
        )

    # Load the base model — training starts from ImageNet pretrained weights
    model = YOLO(str(BASE_MODEL))

    results = model.train(
        # --- Dataset ---
        # Points to training/data.yaml which sets paths relative to data/ directory.
        # train = data/images/train (310 images: 240 annotated + 70 background)
        # val   = data/images/val   (45 images)
        data=str(DATA_YAML),

        # --- Training duration ---
        # 300 epochs max — small dataset converges fast, patience handles early stop.
        # WHY 300: gives enough room to find the precision peak without manual watching.
        epochs=300,

        # Stop early if val mAP doesn't improve for 60 consecutive epochs.
        # WHY 60 (not default 50): small datasets plateau and then recover — give it time.
        patience=60,

        # --- Image resolution ---
        # 1024px is CRITICAL for this dataset.
        # WHY: PCB defects like Spur and Mouse_bite are often 5–20px at 640px resolution,
        # which is below YOLO's reliable detection threshold. 1024px doubles effective
        # defect size and significantly improves small-object recall.
        imgsz=1024,

        # Batch 8 fits a ~8GB GPU with 1024px images.
        # WHY NOT larger: 1024px images are ~4x memory cost vs 640px.
        # Reduce to 4 if you get CUDA OOM errors.
        batch=8,

        # Number of CPU workers for data loading.
        # WHY 2 (not 4+): OneDrive path can cause I/O bottlenecks with high worker count.
        workers=2,

        # Use GPU 0. Change to device='cpu' if no GPU is available.
        # WHY GPU: 1024px images + 300 epochs on CPU would take days.
        device=0,

        # --- Output paths ---
        # Saves to: runs/detect/pcb_precision_v1/
        # best.pt will be at: runs/detect/pcb_precision_v1/weights/best.pt
        # This path is used by training/scripts/mine_hard_negatives.py (line 22)
        # and should later be copied to weights/best.pt for app/model/predictor.py
        project=str(PROJECT_ROOT / "runs" / "detect"),
        name="pcb_precision_v1",

        # exist_ok=False means training will FAIL if pcb_precision_v1 folder already exists.
        # WHY: prevents accidentally overwriting a good training run.
        # Change to True only if you want to resume an interrupted run.
        exist_ok=False,

        # Save a checkpoint .pt file every 50 epochs as a safety net.
        save_period=50,

        # Generate training plots (loss curves, P-R curves, confusion matrix).
        # Saved to runs/detect/pcb_precision_v1/. Useful for diagnosing overfitting.
        plots=True,
        val=True,

        # --- Optimizer ---
        # AdamW over SGD because:
        # WHY: small dataset + AdamW converges faster and more stably than SGD.
        # SGD is better for large datasets with many iterations; AdamW for small ones.
        optimizer="AdamW",

        # Initial learning rate 0.005 — lower than default (0.01).
        # WHY: small dataset (310 images) overfits quickly with high LR.
        # Lower LR = smaller weight updates = more stable convergence.
        lr0=0.005,

        # Final LR = lr0 * lrf = 0.005 * 0.01 = 0.00005 at last epoch.
        # WHY: cosine decay to near-zero — prevents LR from staying too high late in training.
        lrf=0.01,

        momentum=0.937,      # Standard AdamW momentum, no reason to change.
        weight_decay=0.0005, # L2 regularization — prevents overfitting on small dataset.

        # 5 epoch warmup — LR starts near 0 and ramps up to lr0 over 5 epochs.
        # WHY: without warmup, early batches with random weights + full LR cause
        # unstable loss spikes that damage the pretrained backbone features.
        warmup_epochs=5,

        # --- Loss weights ---
        # box=7.5: default, controls bounding box regression loss. No change needed.
        box=7.5,

        # cls=2.0 — RAISED from default 0.5. THIS IS THE KEY PRECISION FIX.
        # WHY: cls loss penalizes wrong class predictions. At default 0.5, the model
        # learns "detect something" more than "classify correctly", causing false positives
        # where it predicts defects on clean regions. Raising to 2.0 forces the model
        # to be more certain before predicting any class → fewer false positives.
        cls=2.0,

        dfl=1.5, # Distribution Focal Loss — controls anchor-free box prediction. Default OK.

        # --- Augmentation (conservative for PCB) ---
        # WHY conservative: PCBs are manufactured objects with fixed geometry and consistent
        # industrial lighting. Aggressive augmentation creates unrealistic training samples
        # that confuse the model more than they help.

        hsv_h=0.01,   # Minimal hue shift — PCB color under factory lighting is fixed.
                      # More shift = model sees impossible color combinations.
        hsv_s=0.3,    # Moderate saturation — handles camera exposure variation.
        hsv_v=0.3,    # Moderate brightness — handles lighting intensity variation.

        degrees=5.0,  # Small rotation only — PCBs are placed consistently on conveyor.
                      # Large rotation (>10 deg) creates impossible factory-floor samples.

        translate=0.05, # Minimal translation — keeps defects near image center.
        scale=0.3,      # Moderate scale — simulates different camera distances.

        shear=0.0,      # OFF — shear distorts PCB geometry, creates fake defect shapes.
        perspective=0.0,# OFF — perspective warp makes rectangular PCBs look trapezoidal.
        flipud=0.0,     # OFF — PCBs have orientation (top/bottom matter for defect context).
        fliplr=0.5,     # ON — horizontal flip is safe, PCBs can be placed either way.

        # Mosaic combines 4 images into 1 during training.
        # WHY 0.5 (not default 1.0): at 1.0, mosaic stitches together different defect
        # classes creating impossible combinations (e.g., Mouse_bite next to Open_circuit
        # on the same solder joint). Reduces this but keeps some variety benefit.
        mosaic=0.5,

        mixup=0.0,      # OFF — blends two images together. Creates ghost defect boundaries.
                        # Destroys the sharp visual boundaries that distinguish PCB defects.
        copy_paste=0.0, # OFF — copy-paste segmentation augmentation. Not useful for YOLO bbox.

        # Disable mosaic for the LAST 20 epochs.
        # WHY: final epochs should train on clean, unaugmented samples so the model
        # learns to predict on real single images, not mosaic composites.
        # This improves final validation metrics.
        close_mosaic=20,

        # --- Validation inference settings ---
        # conf=0.001 during training validation captures the FULL precision-recall curve.
        # WHY: if conf were 0.5, we'd only see metrics at one operating point.
        # At 0.001 we see all thresholds → best.pt is saved at the best mAP point.
        conf=0.001,

        # IoU threshold for NMS during validation.
        # 0.5 = standard COCO threshold. Fine for non-overlapping PCB defects.
        iou=0.5,

        # Max 50 detections per image — a real PCB will never have 300+ defects.
        # WHY: default is 300 (set for COCO crowded scenes). Reducing it avoids
        # wasting compute on spurious low-confidence detections.
        max_det=50,

        rect=False, # Rectangular training OFF — keeps square aspect ratio.
                    # WHY: rect=True changes image padding per batch, which can
                    # cause inconsistent feature maps on a small dataset.

        # --- Performance ---
        # Cache images in RAM for faster training.
        # WHY: with only 310 images, they fit easily in RAM. Eliminates disk I/O
        # bottleneck on repeated epoch reads (especially on OneDrive path).
        cache=True,

        # Automatic Mixed Precision (FP16 + FP32).
        # WHY: halves GPU memory usage, ~1.5x faster training, no accuracy loss.
        amp=True,

        # Light dropout (10%) in the classifier head only.
        # WHY: with 310 images the classifier head can overfit to training patterns.
        # Dropout randomly drops 10% of neurons per forward pass → forces generalization.
        dropout=0.1,

        verbose=True, # Print per-epoch metrics to console. Keep ON to monitor training.
    )

    # --- Results ---
    # best.pt is saved to: runs/detect/pcb_precision_v1/weights/best.pt
    # After verifying metrics, copy it to weights/best.pt so that:
    #   - app/model/predictor.py (line 20) picks it up for inference
    #   - training/scripts/mine_hard_negatives.py (line 22) uses it for Phase 3
    print("\n=== Training Complete ===")
    print(f"Best weights: {PROJECT_ROOT}/runs/detect/pcb_precision_v1/weights/best.pt")

    # Metrics printed here are from the BEST epoch (highest mAP50), not last epoch.
    # Target after Phase 2: Precision >= 0.65, mAP50 >= 0.50
    # If precision < 0.55 → run mine_hard_negatives.py then retrain as pcb_precision_v2
    metrics = results.results_dict
    print(f"Precision:  {metrics.get('metrics/precision(B)', 0):.4f}")
    print(f"Recall:     {metrics.get('metrics/recall(B)', 0):.4f}")
    print(f"mAP50:      {metrics.get('metrics/mAP50(B)', 0):.4f}")
    print(f"mAP50-95:   {metrics.get('metrics/mAP50-95(B)', 0):.4f}")


if __name__ == "__main__":
    main()
