#!/usr/bin/env python3
"""
finish_run.py - Resume an interrupted benchmark and run it through to evaluation.

Idempotent: for each model, in order, it
  * SKIPS runs that already reached the target epoch count,
  * RESUMES a run that has weights/last.pt but did not finish (e.g. yolov8m
    when the network dropped),
  * TRAINS fresh any model that hasn't started,
then runs evaluate.py at the end. Safe to re-run after any future interruption.

Usage (from the repo root):
    PYTHONPATH=src python finish_run.py                 # epochs=300 (default)
    PYTHONPATH=src python finish_run.py --epochs 300 --batch 32 --device 0
    PYTHONPATH=src python finish_run.py --no-eval       # skip the final evaluation
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ultralytics import YOLO
import train as T   # reuse the recipe, paths, and per-model train functions

# (run_name, kind) in the SAME order train.py --model all uses.
ORDER = [
    ("yolov8n", "baseline"),
    ("yolov8s", "baseline"),
    ("yolov8m", "baseline"),
    ("yolov8l", "baseline"),
    ("yolov8x", "baseline"),
    ("efficientvit_yolov8", "efficientvit"),
    ("mobilevit_yolov8", "mobilevit"),
    # EfficientFormerV2-S2 is intentionally NOT in the default pipeline: its
    # resolution-coupled attention is incompatible with 640x640 detection in
    # current timm (token-count vs attention-bias mismatch). Run it manually
    # only if you adopt a supported resolution. See README "Known limitations".
]


def last_epoch(name: str) -> int:
    """Last epoch index recorded in results.csv (-1 if none)."""
    csv = T.RESULTS_DIR / name / "results.csv"
    if not csv.exists():
        return -1
    rows = [r for r in csv.read_text().strip().splitlines() if r.strip()]
    if len(rows) < 2:
        return -1
    try:
        return int(float(rows[-1].split(",")[0]))
    except Exception:
        return -1


def valid_ckpt(p: Path) -> bool:
    """A .pt is a zip; a truncated/corrupt checkpoint fails this cheap check."""
    return p.exists() and p.stat().st_size > 0 and zipfile.is_zipfile(p)


def is_resumable(p: Path) -> bool:
    """True only if the checkpoint still carries optimizer + epoch state.
    A *completed* run has these stripped, so this is the reliable
    interrupted-vs-finished signal (and avoids the destructive case where
    resume() on a stripped checkpoint silently restarts default training)."""
    if not valid_ckpt(p):
        return False
    try:
        ck = torch.load(str(p), map_location="cpu", weights_only=False)
    except Exception:
        return False
    return ck.get("optimizer") is not None and int(ck.get("epoch", -1)) >= 0


def newest_resumable_epoch(wd: Path):
    """Newest intact, resumable epoch*.pt (used to heal a corrupt last.pt)."""
    def epnum(p):
        m = re.findall(r"\d+", p.stem)
        return int(m[0]) if m else -1
    for p in sorted(wd.glob("epoch*.pt"), key=epnum, reverse=True):
        if is_resumable(p):
            return p
    return None


def decide(run: str, epochs: int):
    """Return ('done'|'resume'|'fresh', checkpoint_or_None)."""
    wd = T.RESULTS_DIR / run / "weights"
    best, last = wd / "best.pt", wd / "last.pt"
    if last_epoch(run) >= epochs:
        return "done", None                      # reached target epochs
    if is_resumable(last):
        return "resume", last                    # genuinely interrupted
    if valid_ckpt(last) and best.exists():
        return "done", None                      # stripped last + best => finished (early stop)
    ep = newest_resumable_epoch(wd)              # last.pt corrupt/missing -> try periodic
    if ep is not None:
        return "resume", ep
    if best.exists():
        return "done", None                      # something finished; never destroy it
    return "fresh", None


def register(kind: str):
    if kind == "efficientvit":
        from models.efficientvit_modules import register_efficientvit_modules
        register_efficientvit_modules()
    elif kind in ("mobilevit", "efficientformer"):
        from models.timm_backbone import register_timm_backbone_modules
        register_timm_backbone_modules()


def train_fresh(run, kind, kw):
    if kind == "baseline":
        T.train_yolov8_baseline(run, **kw)          # run name == variant
    elif kind == "efficientvit":
        T.train_efficientvit(**kw)
    elif kind == "mobilevit":
        T.train_timm("mobilevit", **kw)
    elif kind == "efficientformer":
        T.train_timm("efficientformer", **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300, help="target epochs of the original run")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--data", default=str(T.DEFAULT_DATA))
    ap.add_argument("--project", default=None,
                    help="output dir for runs (default ./results_acf).")
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    if args.project:                       # redirect all run I/O to an isolated dir
        T.RESULTS_DIR = Path(args.project).resolve()
    if not Path(args.data).exists():
        raise SystemExit(f"data.yaml not found: {args.data}\nRun the matching prepare_*.py first.")
    T.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    kw = dict(data=args.data, epochs=args.epochs, batch=args.batch,
              imgsz=args.imgsz, device=args.device)

    # NB: do NOT pre-register the custom modules here. torch.load() only needs
    # the model classes importable (src is on sys.path), not the parse_model
    # patch; and registering EfficientViT then MobileViT in one process corrupts
    # inspect.getsource for the second. Each model registers itself when trained.

    # ---- plan (decide once, reuse) ----
    plan = {run: decide(run, args.epochs) for run, _ in ORDER}
    print("Plan:")
    for run, _ in ORDER:
        state, ckpt = plan[run]
        extra = f" from {ckpt.name} (epoch {last_epoch(run)})" if state == "resume" else ""
        print(f"  {run:22} {state.upper()}{extra}")
    print()

    # ---- execute ----
    for run, kind in ORDER:
        state, ckpt = plan[run]
        if state == "done":
            print(f"[skip] {run}: already finished")
            continue
        if state == "resume":
            print(f"[resume] {run} from {ckpt.name}")
            register(kind)
            try:
                YOLO(str(ckpt)).train(resume=True)
            except Exception as e:
                print(f"[warn] resume of {run} failed ({e}); retraining fresh")
                train_fresh(run, kind, kw)
        else:
            print(f"[train] {run} (fresh)")
            train_fresh(run, kind, kw)

    if args.no_eval:
        print("[done] training complete; --no-eval set, skipping evaluation.")
        return

    print("\n[eval] running evaluate.py ...")
    env_cmd = [sys.executable, str(ROOT / "evaluate.py"),
               "--data", args.data, "--imgsz", str(args.imgsz), "--device", args.device]
    if args.project:
        env_cmd += ["--project", str(T.RESULTS_DIR)]
    subprocess.run(env_cmd, cwd=str(ROOT), check=False)


if __name__ == "__main__":
    main()
