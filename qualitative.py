#!/usr/bin/env python3
"""
qualitative.py - Side-by-side qualitative comparison of the trained backbones.

For a set of CCTV test frames it runs every trained model, draws the predicted
boxes, and saves one comparison panel per image (Ground Truth + each model) into
results_acf/qualitative/. Each panel also carries a magnified inset of the weapon
region (derived from the ground-truth box) so the small CCTV targets are legible.
Useful as a publication figure.

Usage (from repo root):
    PYTHONPATH=src python qualitative.py --n 50 --conf 0.15
    PYTHONPATH=src python qualitative.py --require-class 1 --n 50 --prefix compare_knife
    PYTHONPATH=src python qualitative.py --images Cam7-...frame_14 Cam1-...frame_152
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_DATA_DIR = ROOT / "data" / "ACF"   # CCTV (the only dataset in this repo)
DEFAULT_PROJECT = ROOT / "results_acf"      # CCTV in-domain weights

# Display order (label -> run dir). Skips any that aren't trained.
MODELS = [
    ("YOLOv8n", "yolov8n", None),
    ("YOLOv8s", "yolov8s", None),
    ("YOLOv8m", "yolov8m", None),
    ("YOLOv8l", "yolov8l", None),
    ("YOLOv8x", "yolov8x", None),
    ("EfficientViT-B1", "efficientvit_yolov8", "efficientvit"),
    ("MobileViT-S", "mobilevit_yolov8", "timm"),
]


def register(kind):
    try:
        if kind == "efficientvit":
            from models.efficientvit_modules import register_efficientvit_modules
            register_efficientvit_modules()
        elif kind == "timm":
            from models.timm_backbone import register_timm_backbone_modules
            register_timm_backbone_modules()
    except Exception as e:
        print(f"[warn] registration for '{kind}' skipped ({e})")


def load_gt(img_path: Path, label_dir: Path):
    """Return list of (cls, x1,y1,x2,y2) in pixel coords from a YOLO label file."""
    lp = label_dir / f"{img_path.stem}.txt"
    if not lp.exists():
        return []
    w, h = Image.open(img_path).size
    out = []
    for line in lp.read_text().strip().splitlines():
        if not line.strip():
            continue
        c, xc, yc, bw, bh = (float(v) for v in line.split())
        out.append((int(c), (xc - bw / 2) * w, (yc - bh / 2) * h,
                    (xc + bw / 2) * w, (yc + bh / 2) * h))
    return out


NAMES = {0: "Handgun", 1: "Knife"}
COLORS = {0: (220, 40, 40), 1: (40, 90, 220)}

_FONT_PATHS = [
    "DejaVuSans-Bold.ttf", "DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def _font(size: int):
    for p in _FONT_PATHS:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


ZOOM_BORDER = (255, 215, 0)   # gold frame linking ROI box <-> magnified inset


def weapon_roi(boxes, size, pad_frac=1.2, min_frac=0.16):
    """Square region to magnify, from the union of (GT) boxes. Returns
    (x1,y1,x2,y2) or None if there is no box to zoom into."""
    if not boxes:
        return None
    W, H = size
    xs1 = min(b[1] for b in boxes); ys1 = min(b[2] for b in boxes)
    xs2 = max(b[3] for b in boxes); ys2 = max(b[4] for b in boxes)
    cx, cy = (xs1 + xs2) / 2, (ys1 + ys2) / 2
    # half-size: pad the box, but never smaller than min_frac of the short side
    half = max((xs2 - xs1) * (1 + pad_frac) / 2,
               (ys2 - ys1) * (1 + pad_frac) / 2,
               min(W, H) * min_frac / 2)
    half = min(half, min(W, H) / 2)            # keep it on-frame
    x1 = max(0, cx - half); y1 = max(0, cy - half)
    x2 = min(W, cx + half); y2 = min(H, cy + half)
    return (int(x1), int(y1), int(x2), int(y2))


def annotate(img: Image.Image, boxes, title: str, roi=None) -> Image.Image:
    im = img.convert("RGB").copy()
    W, H = im.size
    # Scale fonts/strokes to image width so the model label is clearly readable
    # even on 1920px CCTV frames.
    title_fs = max(28, W // 22)
    box_fs = max(20, W // 32)
    lw = max(3, W // 300)
    bar_h = int(title_fs * 1.7)
    tfont, bfont = _font(title_fs), _font(box_fs)

    d = ImageDraw.Draw(im)
    for c, x1, y1, x2, y2 in boxes:
        col = COLORS.get(int(c), (0, 160, 0))
        d.rectangle([x1, y1, x2, y2], outline=col, width=lw)
        label = NAMES.get(int(c), str(c))
        ty = max(0, y1 - box_fs - 6)
        tw = d.textlength(label, font=bfont)
        d.rectangle([x1, ty, x1 + tw + 10, ty + box_fs + 6], fill=col)
        d.text((x1 + 5, ty + 2), label, fill=(255, 255, 255), font=bfont)

    # --- magnified inset of the weapon region ---
    if roi is not None:
        rx1, ry1, rx2, ry2 = roi
        rw, rh = max(1, rx2 - rx1), max(1, ry2 - ry1)
        crop = img.convert("RGB").crop((rx1, ry1, rx2, ry2))   # clean (box-free) crop
        iw = max(180, W // 3)                                   # inset ~1/3 of frame width
        ih = int(iw * rh / rw)
        big = crop.resize((iw, ih), Image.LANCZOS)
        dd = ImageDraw.Draw(big)
        ilw = max(2, iw // 90)
        sx, sy = iw / rw, ih / rh
        for c, x1, y1, x2, y2 in boxes:                        # redraw boxes at inset scale
            ix1, iy1 = max(0, x1 - rx1), max(0, y1 - ry1)
            ix2, iy2 = min(rw, x2 - rx1), min(rh, y2 - ry1)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            col = COLORS.get(int(c), (0, 160, 0))
            dd.rectangle([ix1 * sx, iy1 * sy, ix2 * sx, iy2 * sy], outline=col, width=ilw)
        dd.rectangle([0, 0, iw - 1, ih - 1], outline=ZOOM_BORDER, width=max(3, iw // 80))
        # box on the full frame showing where the inset came from
        d.rectangle([rx1, ry1, rx2, ry2], outline=ZOOM_BORDER, width=lw)
        # paste in the bottom corner away from the weapon
        pad = max(6, W // 160)
        left_half = (rx1 + rx2) / 2 < W / 2
        px = (W - iw - pad) if left_half else pad              # opposite side to the ROI
        py = H - ih - pad
        im.paste(big, (px, py))

    bar = Image.new("RGB", (W, bar_h), (245, 245, 245))
    ImageDraw.Draw(bar).text((10, (bar_h - title_fs) // 2), title, fill=(0, 0, 0), font=tfont)
    out = Image.new("RGB", (W, im.height + bar_h), (255, 255, 255))
    out.paste(bar, (0, 0)); out.paste(im, (0, bar_h))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=4, help="number of auto-picked images")
    ap.add_argument("--images", nargs="*", default=None,
                    help="specific image stems (without extension) to use")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                    help="dataset root with <split>/images and <split>/labels "
                         "(default data/ACF; use data/Sohas for the Sohas set)")
    ap.add_argument("--project", default=str(DEFAULT_PROJECT),
                    help="results dir holding <model>/weights/best.pt "
                         "(default results_acf, the CCTV in-domain models)")
    ap.add_argument("--seed", type=int, default=0, help="selection seed for --n")
    ap.add_argument("--require-class", type=int, default=None,
                    help="only auto-pick frames whose GT contains this class id "
                         "(0=Handgun, 1=Knife). Use to get knife-only examples.")
    ap.add_argument("--prefix", default="compare",
                    help="output filename prefix (e.g. 'compare_knife' to keep "
                         "knife strips separate from the handgun set)")
    args = ap.parse_args()

    results_dir = Path(args.project).resolve()
    out_dir = results_dir / "qualitative"
    data_dir = Path(args.data_dir)
    img_dir = data_dir / args.split / "images"
    lbl_dir = data_dir / args.split / "labels"
    if not img_dir.exists():
        raise SystemExit(f"{img_dir} not found. Run the matching prepare_*.py first.")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.images:
        sel = [img_dir / s if (img_dir / s).exists() else next((p for p in all_imgs if p.stem == s), None)
               for s in args.images]
        sel = [p for p in sel if p]
    elif args.require_class is not None:
        # only frames whose GT contains the requested class (e.g. 1=Knife).
        pool = [p for p in all_imgs
                if any(int(b[0]) == args.require_class for b in load_gt(p, lbl_dir))]
        cname = NAMES.get(args.require_class, str(args.require_class))
        print(f"[i] {len(pool)} frames contain class {args.require_class} ({cname})")
        if not pool:
            raise SystemExit(
                f"No '{cname}' frames in {lbl_dir}. This dataset may not contain "
                f"that class (CCTV is gun-focused; knives are in data/Sohas).")
        random.Random(args.seed).shuffle(pool)
        sel = pool[: args.n]
    else:
        # auto-pick: prefer images with at least one GT box (a real weapon),
        # shuffled (seeded) so a large --n gives a varied set to choose from.
        with_gt = [p for p in all_imgs if load_gt(p, lbl_dir)]
        pool = (with_gt or all_imgs)[:]
        random.Random(args.seed).shuffle(pool)
        sel = pool[: args.n]
    if not sel:
        raise SystemExit("No images selected.")
    print(f"[i] {len(sel)} images selected")

    # Load available models once.
    loaded = []
    for label, run, kind in MODELS:
        ckpt = results_dir / run / "weights" / "best.pt"
        if not ckpt.exists():
            print(f"[skip] {label}: no checkpoint")
            continue
        if kind:
            register(kind)
        loaded.append((label, YOLO(str(ckpt))))
    if not loaded:
        raise SystemExit("No trained models found.")

    for img_path in sel:
        gt = load_gt(img_path, lbl_dir)
        with Image.open(img_path) as _im:
            roi = weapon_roi(gt, _im.size)   # same zoom region for every panel
        panels = [annotate(Image.open(img_path), gt, "Ground Truth", roi=roi)]
        for label, model in loaded:
            r = model.predict(source=str(img_path), imgsz=args.imgsz, conf=args.conf,
                              device=args.device, verbose=False)[0]
            boxes = [(int(b.cls.item()), *b.xyxy[0].tolist()) for b in r.boxes]
            panels.append(annotate(Image.open(img_path), boxes, label, roi=roi))
        # stack horizontally (uniform height)
        h = max(p.height for p in panels)
        widths = [int(p.width * h / p.height) for p in panels]
        strip = Image.new("RGB", (sum(widths) + 6 * (len(panels) - 1), h), (255, 255, 255))
        x = 0
        for p, w in zip(panels, widths):
            strip.paste(p.resize((w, h)), (x, 0)); x += w + 6
        out = out_dir / f"{args.prefix}_{img_path.stem}.png"
        strip.save(out)
        print(f"[ok] {out}")

    print(f"\n[done] {len(sel)} panels in {out_dir}")


if __name__ == "__main__":
    main()
