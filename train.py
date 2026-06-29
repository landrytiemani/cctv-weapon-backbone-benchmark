#!/usr/bin/env python3
"""
train.py - Seven-backbone weapon-detection benchmark on the public US real-time
CCTV gun-detection dataset (ACF).

Trains YOLOv8 (n/s/m/l/x) and the edge ViT backbones (EfficientViT-B1,
MobileViT-S) integrated into the YOLOv8 detection framework, under one identical
SGD recipe so the backbone is the only varying component.

Run prepare_acf.py first to build data/ACF/.

Usage (from the repository root):
    PYTHONPATH=src python train.py --model all
    PYTHONPATH=src python train.py --model yolov8x
    PYTHONPATH=src python train.py --model efficientvit
    PYTHONPATH=src python train.py --model mobilevit
    PYTHONPATH=src python train.py --model all --data /abs/data.yaml
    # quick end-to-end smoke test (tiny, just checks the pipeline runs):
    PYTHONPATH=src python train.py --model yolov8n --smoke
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent          # standalone repo root
sys.path.insert(0, str(ROOT / "src"))

CONFIGS_DIR = ROOT / "configs"
DEFAULT_DATA = ROOT / "data" / "ACF" / "data.yaml"
RESULTS_DIR = ROOT / "results_acf"

EFFICIENTVIT_CONFIG = str(CONFIGS_DIR / "efficientvit_yolov8.yaml")   # B1
TIMM_CONFIGS = {
    "mobilevit": str(CONFIGS_DIR / "mobilevit_yolov8.yaml"),
    "efficientformer": str(CONFIGS_DIR / "efficientformer_yolov8.yaml"),
}

# --- Paper-matched training recipe (Berardini et al. MESA 2024 protocol) -------
# Same recipe for every detector so the comparison is fair:
#   SGD, lr0=0.05, momentum=0.9, weight_decay=5e-4, batch=32, 300 epochs,
#   linear LR decay to lrf=0.01, patience=100, imgsz=640,
#   augment = mosaic + HSV + hflip + translate + scale  (NO mixup/copy_paste).
PAPER_RECIPE = dict(
    optimizer="SGD", lr0=0.05, lrf=0.01, momentum=0.9, weight_decay=0.0005,
    warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, cos_lr=False,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, flipud=0.0, fliplr=0.5,
    mosaic=1.0, mixup=0.0, copy_paste=0.0, translate=0.1, scale=0.5,
    close_mosaic=10, box=7.5, cls=0.5, dfl=1.5,
    patience=100, save=True, save_period=50, verbose=True,
)

BASELINES = ("yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x")


def _train(model, name, data, epochs, batch, imgsz, device, **extra):
    return model.train(data=data, epochs=epochs, batch=batch, imgsz=imgsz,
                       device=device, project=str(RESULTS_DIR), name=name,
                       exist_ok=True, **{**PAPER_RECIPE, **extra})


def train_yolov8_baseline(variant, **kw):
    print(f"\n=== YOLOv8 baseline: {variant} ===")
    return _train(YOLO(f"{variant}.pt"), variant, **kw)


def train_efficientvit(**kw):
    print("\n=== EfficientViT-B1 + YOLOv8 (ImageNet backbone, scratch neck/head) ===")
    from models.efficientvit_modules import register_efficientvit_modules
    from models.pretrained_init import load_all_pretrained
    register_efficientvit_modules()
    model = YOLO(EFFICIENTVIT_CONFIG, task="detect")
    load_all_pretrained(model.model, load_neck=False)
    return _train(model, "efficientvit_yolov8", pretrained=False, amp=False,
                  warmup_epochs=10, **kw)


def train_timm(family, **kw):
    print(f"\n=== {family} + YOLOv8 (timm backbone, scratch neck/head) ===")
    from models.timm_backbone import register_timm_backbone_modules
    register_timm_backbone_modules()
    model = YOLO(TIMM_CONFIGS[family], task="detect")
    return _train(model, f"{family}_yolov8", pretrained=False, amp=False,
                  warmup_epochs=10, **kw)


def main():
    ap = argparse.ArgumentParser(description="Train the 7-backbone benchmark on the CCTV (ACF) dataset")
    ap.add_argument("--model", default="all",
                    choices=[*BASELINES, "efficientvit", "mobilevit",
                             "efficientformer", "all"])
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--project", default=None,
                    help="output dir for runs (default ./results_acf).")
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny run (2 epochs, batch 4) to validate the pipeline")
    args = ap.parse_args()

    global RESULTS_DIR
    if args.project:
        RESULTS_DIR = Path(args.project).resolve()
    if not Path(args.data).exists():
        raise SystemExit(f"data.yaml not found: {args.data}\nRun the matching prepare_*.py first.")
    if args.smoke:
        args.epochs, args.batch = 2, 4
        print("[smoke] epochs=2 batch=4")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    kw = dict(data=args.data, epochs=args.epochs, batch=args.batch,
              imgsz=args.imgsz, device=args.device)
    sel = args.model

    for v in BASELINES:
        if sel in (v, "all"):
            train_yolov8_baseline(v, **kw)
    if sel in ("efficientvit", "all"):
        train_efficientvit(**kw)
    if sel in ("mobilevit", "all"):
        train_timm("mobilevit", **kw)
    if sel == "efficientformer":          # optional; not in the headline 7
        train_timm("efficientformer", **kw)


if __name__ == "__main__":
    main()
