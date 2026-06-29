#!/usr/bin/env python3
"""
evaluate.py - Evaluate the trained seven-backbone benchmark on the CCTV (ACF)
test split and write one comparison table (CSV + JSON): mean AP50, AP50:95,
per-class AP50 / AP50:95 (Handgun, Knife), Precision, Recall, F1, single-image
FPS (measured on the eval GPU after a warm-up), Params (M), GFLOPs.

FPS is hardware-specific: it is measured here on the same GPU used for
evaluation, at batch size 1, end-to-end (preprocess + inference + postprocess),
after a warm-up. Report the GPU used alongside the number.

Usage (from the repository root):
    PYTHONPATH=src python evaluate.py
    PYTHONPATH=src python evaluate.py --no-fps          # skip FPS timing
    PYTHONPATH=src python evaluate.py --fps-iters 300 --fps-warmup 50
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_DATA = ROOT / "data" / "ACF" / "data.yaml"
RESULTS_DIR = ROOT / "results_acf"

# class id -> name (must match data.yaml: 0 Handgun, 1 Knife)
NAMES = {0: "Handgun", 1: "Knife"}

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


def per_class_ap(m, names=NAMES):
    """Return ({cid: AP50%}, {cid: AP50:95%}) per class from a val() result.
    Classes with no detections (absent from ap_class_index) stay None."""
    ap50 = {c: None for c in names}
    ap5095 = {c: None for c in names}
    try:
        idx = [int(c) for c in m.box.ap_class_index]      # classes that have results
        ap50_arr = m.box.ap50                              # AP@0.5 aligned to idx
        for j, c in enumerate(idx):
            if c in ap50:
                ap50[c] = round(float(ap50_arr[j]) * 100, 2)
        maps = getattr(m.box, "maps", None)                # per-class mAP@0.5:0.95, indexed by class id
        if maps is not None:
            for c in names:
                try:
                    ap5095[c] = round(float(maps[c]) * 100, 2)
                except Exception:
                    pass
    except Exception as e:
        print(f"[warn] per-class metrics unavailable ({e})")
    return ap50, ap5095


def measure_fps(model, imgsz, device, warmup=50, iters=200):
    """Single-image, end-to-end FPS on the eval GPU after a warm-up. Uses a
    blank frame so the number reflects compute, not image I/O; batch size 1."""
    frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    try:
        import torch
        cuda = (str(device) != "cpu") and torch.cuda.is_available()
    except Exception:
        torch, cuda = None, False
    for _ in range(warmup):
        model.predict(source=frame, imgsz=imgsz, device=device, verbose=False)
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model.predict(source=frame, imgsz=imgsz, device=device, verbose=False)
    if cuda:
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return round(iters / dt, 1) if dt > 0 else None


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
    ap.add_argument("--no-fps", action="store_true", help="skip the FPS timing loop")
    ap.add_argument("--fps-warmup", type=int, default=50, help="FPS warm-up iterations")
    ap.add_argument("--fps-iters", type=int, default=200, help="FPS timed iterations")
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
        ap50_pc, ap5095_pc = per_class_ap(m, NAMES)
        fps = None if args.no_fps else measure_fps(
            model, args.imgsz, args.device, args.fps_warmup, args.fps_iters)
        rows.append({
            "model": run,
            "AP50": round(float(m.box.map50) * 100, 2),
            "AP50_95": round(float(m.box.map) * 100, 2),
            "AP50_Handgun": ap50_pc[0],
            "AP50_Knife": ap50_pc[1],
            "AP5095_Handgun": ap5095_pc[0],
            "AP5095_Knife": ap5095_pc[1],
            "Precision": round(p * 100, 2),
            "Recall": round(r * 100, 2),
            "F1": round(f1 * 100, 2),
            "FPS": fps,
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

    hdr = (f'{"model":24}{"AP50":>7}{"AP50:95":>9}{"AP50_H":>8}{"AP50_K":>8}'
           f'{"F1":>7}{"FPS":>7}{"Params":>8}{"GFLOPs":>8}')
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for x in rows:
        print(f'{x["model"]:24}{x["AP50"]:7}{x["AP50_95"]:9}'
              f'{str(x["AP50_Handgun"]):>8}{str(x["AP50_Knife"]):>8}'
              f'{x["F1"]:7}{str(x["FPS"]):>7}{str(x["Params_M"]):>8}{str(x["GFLOPs"]):>8}')
    print(f'\n[OK] wrote {RESULTS_DIR}/{tag}_comparison.csv and .json')
    print("[i] FPS = single-image end-to-end throughput at batch 1 on the eval GPU "
          "(report the GPU model alongside it).")


if __name__ == "__main__":
    main()
