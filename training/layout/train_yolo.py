"""Fine-tune YOLOv8 on exam-paper layouts.

Starting weights: a DocLayNet-pretrained YOLOv8 checkpoint if you have one (best),
otherwise COCO yolov8s.pt. Augmentations are mild and document-safe: no flips (text has
a direction), small rotations only, so the model learns real paper geometry.

  python training/layout/train_yolo.py --data training/datasets/exam_layout/data.yaml
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "training/datasets/exam_layout/data.yaml"))
    ap.add_argument("--model", default="yolov8s.pt", help="or a DocLayNet-pretrained .pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None, help="0 for GPU, cpu for CPU")
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.model)
    model.train(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
        project=str(ROOT / "training/runs"), name="layout", exist_ok=True,
        patience=20, cos_lr=True,
        fliplr=0.0, flipud=0.0, mosaic=0.3, mixup=0.0,     # documents must not be mirrored
        degrees=2.0, translate=0.05, scale=0.25, shear=0.0, perspective=0.0005,
        hsv_h=0.0, hsv_s=0.2, hsv_v=0.3,                   # lighting changes like phone photos
    )
    best = ROOT / "training/runs/layout/weights/best.pt"
    target = ROOT / "models/layout_yolov8.pt"
    target.parent.mkdir(exist_ok=True)
    shutil.copy2(best, target)
    metrics = YOLO(str(target)).val(data=a.data, imgsz=a.imgsz)
    print(f"Saved {target}  mAP50-95={metrics.box.map:.3f}  mAP50={metrics.box.map50:.3f}")
    print("Set layout.engine: yolo in configs/default.yaml to use it.")


if __name__ == "__main__":
    main()
