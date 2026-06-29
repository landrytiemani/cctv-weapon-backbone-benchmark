#!/usr/bin/env python3
"""
evaluate.py - Evaluate the trained seven-backbone benchmark on the CCTV (ACF)
test split and write one comparison table (CSV + JSON): AP50, AP50:95,
Precision, Recall, F1, Params (M), GFLOPs.

Edge-latency (Jetson-Nano FPS) is dataset-independent; report it from your
device measurements - it does not change with the dataset.

Usage (from the repository root):
    PYTHONPATH=src python evaluate.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_DATA = ROOT / "data" / "ACF" / "data.yaml"
RESULTS_DIR = ROOT / "results_acf"

# run-name -> custom-module kind needed to load the checkpoint
RUNS = {
    "yolov8n": None, "yolov8s": None, "yolov8m": None,
    "yolov8l": None, "yolov8x": None,
    "efficientvit_yolov8": "efficientvit",
    "mobilevit_yolov8": "timm",
    "efficientformer_yolov8": "timm",
}


def register(kind):
    # Loading a trained checkpoint only needs the module CLASSES importable
    # (src is on sys.path), not the parse_model patch. So registration failures
    # (e.g. a parse_model layout this adapter can't patch) must NOT block eval.
    try:
        if kind == "efficientvit":
            from models.efficientvit_modules import register_efficientvit_modules
            register_efficientvit_modules()
        elif kind == "timm":
            from models.timm_backbone import register_timm_backbone_modules
            register_timm_backbone_modules()
    except Exception as e:
        print(f"[warn] registration for '{kind}' skipped ({e}); checkpoint load usually does not need it")


def complexity(model, imgsz=640):
    """Return (params_M, GFLOPs). Uses Ultralytics' own helpers (same numbers it
    prints in the model summary); falls back to a direct param count."""
    m = getattr(model, "model", model)
    params = gflops = None
    try:
        from ultralytics.utils.torch_utils import get_num_params, get_flops
        params = round(get_num_params(m) / 1e6, 2)
        g = get_flops(m, imgsz)
        gflops = round(float(g), 2) if g else None
    except Exception:
        try:
            params = round(sum(p.numel() for p in m.parameters()) / 1e6, 2)
        except Exception:
            params = None
    return params, gflops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", default=None,
                    help="output filename prefix; defaults to the dataset dir name "
                         "(e.g. 'acf')")
    ap.add_argument("--project", default=None,
                    help="results dir to read checkpoints from / write the CSV into "
                         "(default ./results_acf). Match the dir used at training time.")
    args = ap.parse_args()
    global RESULTS_DIR
    if args.project:
        RESULTS_DIR = Path(args.project).resolve()
    tag = args.tag or Path(args.data).resolve().parent.name.lower()

    rows = []
    for run, kind in RUNS.items():
        ckpt = RESULTS_DIR / run / "weights" / "best.pt"
        if not ckpt.exists():
            print(f"[skip] {run}: no checkpoint at {ckpt}")
            continue
        if kind:
            register(kind)
        print(f"[eval] {run}")
        model = YOLO(str(ckpt))
        m = model.val(data=args.data, split=args.split, imgsz=args.imgsz,
                      device=args.device, verbose=False)
        p, r = float(m.box.mp), float(m.box.mr)
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        params, gflops = complexity(model, args.imgsz)
        rows.append({
            "model": run,
            "AP50": round(float(m.box.map50) * 100, 2),
            "AP50_95": round(float(m.box.map) * 100, 2),
            "Precision": round(p * 100, 2),
            "Recall": round(r * 100, 2),
            "F1": round(f1 * 100, 2),
            "Params_M": params,
            "GFLOPs": gflops,
        })

    if not rows:
        raise SystemExit("No trained models found under results/. Run train.py first.")

    rows.sort(key=lambda x: x["AP50"], reverse=True)
    (RESULTS_DIR / f"{tag}_comparison.json").write_text(json.dumps(rows, indent=2))
    with (RESULTS_DIR / f"{tag}_comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    hdr = f'{"model":24}{"AP50":>7}{"AP50:95":>9}{"P":>7}{"R":>7}{"F1":>7}{"Params":>8}{"GFLOPs":>8}'
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for x in rows:
        print(f'{x["model"]:24}{x["AP50"]:7}{x["AP50_95"]:9}{x["Precision"]:7}'
              f'{x["Recall"]:7}{x["F1"]:7}{str(x["Params_M"]):>8}{str(x["GFLOPs"]):>8}')
    print(f'\n[OK] wrote {RESULTS_DIR}/{tag}_comparison.csv and .json')


if __name__ == "__main__":
    main()
