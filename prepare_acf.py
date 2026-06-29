#!/usr/bin/env python3
"""
prepare_acf.py
==============
Build the benchmark dataset: convert the public **US Real-time gun detection in
CCTV** dataset (Deepknowledge-US / University of Seville, González et al., Neural
Networks 2020 — the "Mock Attack" real-CCTV subset) into the YOLO layout this
repo trains and evaluates on.

Repo / info:
  https://github.com/Deepknowledge-US/US-Real-time-gun-detection-in-CCTV-An-open-problem-dataset
Cite:
  González, J.L.S., Zaccaro, C., Álvarez-García, J.A., Soria-Morillo, L.M.,
  Sancho-Caparrini, F. (2020). Real-time gun detection in CCTV: An open problem.
  Neural Networks 132, 297-308.

Class mapping to this repo's 2 classes:
  pistol / handgun / gun / arma_corta  -> 0 (Handgun)
  knife / cuchillo                     -> 1 (Knife)
  rifle / short rifle / arma_larga / shotgun -> dropped (outside our 2 classes;
      the image is still kept so a spurious detection on a rifle counts as a
      false positive, which is the conservative choice for a handgun/knife test).

Two modes:
  * default        : leakage-safe train/val/test split, grouped by camera/segment
                     so frames of one clip never straddle splits (this is the
                     in-domain benchmark the paper reports).
  * --all-test     : put EVERYTHING into the test split, to evaluate an
                     externally-trained checkpoint with no retraining.

Usage:
  # build the in-domain benchmark split:
  python prepare_acf.py --src /path/to/cctv_mock_attack --out ./data --val-frac 0.15
  #   -> data/ACF/{train,val,test}/{images,labels} + data/ACF/data.yaml
  PYTHONPATH=src python finish_run.py            # train all backbones, then evaluate
"""
from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

CLASS_MAP = {
    "pistol": 0, "handgun": 0, "gun": 0, "arma_corta": 0, "handgun_short": 0,
    "knife": 1, "cuchillo": 1, "knive": 1,
    # dropped (outside our 2-class scheme):
    "rifle": None, "short rifle": None, "short_rifle": None, "arma_larga": None,
    "shotgun": None, "long gun": None, "long_gun": None,
}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def index_files(root: Path, exts):
    out = {}
    if root and root.exists():
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                out.setdefault(p.stem, p)
    return out


def parse_voc(xml_path: Path):
    r = ET.parse(xml_path).getroot()
    size = r.find("size")
    w = float(size.findtext("width")) if size is not None and size.findtext("width") else 0.0
    h = float(size.findtext("height")) if size is not None and size.findtext("height") else 0.0
    objs = []
    for obj in r.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        bb = obj.find("bndbox")
        if bb is None:
            continue
        objs.append((name, float(bb.findtext("xmin")), float(bb.findtext("ymin")),
                     float(bb.findtext("xmax")), float(bb.findtext("ymax"))))
    return int(w), int(h), objs


def to_yolo(w, h, objs, unknown):
    lines, has = [], False
    for name, x1, y1, x2, y2 in objs:
        if name not in CLASS_MAP:
            unknown.add(name); continue
        cid = CLASS_MAP[name]
        if cid is None:
            continue
        if w <= 0 or h <= 0:
            continue
        xc, yc = ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        xc, yc = min(max(xc, 0), 1), min(max(yc, 0), 1)
        bw, bh = min(max(bw, 0), 1), min(max(bh, 0), 1)
        if bw > 0 and bh > 0:
            lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"); has = True
    return lines, has


def find_dir(root, *needles):
    for p in [root, *root.rglob("*")]:
        if p.is_dir() and any(n in p.name.lower() for n in needles):
            return p
    return None


def write_split(items, split_dir: Path):
    (split_dir / "images").mkdir(parents=True, exist_ok=True)
    (split_dir / "labels").mkdir(parents=True, exist_ok=True)
    for stem, img, lines in items:
        shutil.copy2(img, split_dir / "images" / img.name)
        (split_dir / "labels" / f"{stem}.txt").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="Convert US-CCTV (ACF) -> YOLO format")
    ap.add_argument("--src", required=True, type=Path,
                    help="Path to the downloaded CCTV mock-attack folder")
    ap.add_argument("--out", default=Path(__file__).parent / "data", type=Path)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--all-test", action="store_true",
                    help="Put every image in the test split (cross-domain eval)")
    ap.add_argument("--random-split", action="store_true",
                    help="Per-frame random split (LEAKS for video frames). Default "
                         "is a leakage-safe split by camera/segment group.")
    ap.add_argument("--keep-negatives", dest="keep_neg", action="store_true", default=True)
    ap.add_argument("--drop-negatives", dest="keep_neg", action="store_false")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    src = args.src.expanduser().resolve()
    ann_dir = find_dir(src, "annotation", "xml", "label") or src
    img_dir = find_dir(src, "image", "jpeg", "frame") or src
    print(f"[i] annotations: {ann_dir}\n[i] images:      {img_dir}")

    xmls = index_files(ann_dir, {".xml"})
    imgs = index_files(img_dir, IMG_EXTS)
    if not xmls:
        raise SystemExit(f"No VOC .xml annotations found under {ann_dir}. "
                         "If this dataset ships YOLO/COCO labels, adapt this script.")
    print(f"[i] {len(xmls)} annotations, {len(imgs)} images")

    records, unknown = [], set()
    npos = nneg = nskip = 0
    for stem, xml in sorted(xmls.items()):
        img = imgs.get(stem)
        if img is None:
            nskip += 1; continue
        w, h, objs = parse_voc(xml)
        lines, has = to_yolo(w, h, objs, unknown)
        if not has:
            if not args.keep_neg:
                nskip += 1; continue
            nneg += 1
        else:
            npos += 1
        records.append((stem, img, lines))

    if unknown:
        print(f"[!] unmapped object names (ignored): {sorted(unknown)}")
    print(f"[i] usable={len(records)} (positives={npos}, negatives={nneg}, skipped={nskip})")
    if not records:
        raise SystemExit("No usable image/annotation pairs.")

    # Per-class instance counts (so you can see if Knife is too rare to keep).
    cls_count = {0: 0, 1: 0}
    for _, _, lines in records:
        for ln in lines:
            cls_count[int(ln.split()[0])] += 1
    print(f"[i] instances -> Handgun(0)={cls_count[0]}  Knife(1)={cls_count[1]}")
    if cls_count[1] < 50:
        print("[!] Knife is very rare; consider treating this as a single-class "
              "(Handgun) benchmark in the paper.")

    def group_of(stem):
        # video segment id = everything before '_frame_' (avoids temporal leakage)
        return stem.split("_frame_")[0]

    if args.all_test:
        train, val, test = [], [], records
    elif args.random_split:
        recs = records[:]; random.shuffle(recs)
        n_test = int(round(0.15 * len(recs)))
        test, rest = recs[:n_test], recs[n_test:]
        n_val = int(round(args.val_frac * len(rest)))
        val, train = rest[:n_val], rest[n_val:]
        print("[!] random per-frame split requested (may leak across video frames)")
    else:
        groups = {}
        for rec in records:
            groups.setdefault(group_of(rec[0]), []).append(rec)
        gkeys = sorted(groups)
        random.shuffle(gkeys)
        n = len(gkeys)
        n_test = max(1, round(0.15 * n))
        n_val = max(1, round(args.val_frac * n))
        test_g = set(gkeys[:n_test])
        val_g = set(gkeys[n_test:n_test + n_val])
        test = [r for g in test_g for r in groups[g]]
        val = [r for g in val_g for r in groups[g]]
        train = [r for g in gkeys[n_test + n_val:] for r in groups[g]]
        print(f"[i] leakage-safe split by segment: {n} groups "
              f"-> train {len(gkeys) - n_test - n_val} / val {n_val} / test {n_test} groups")
    print(f"[i] split -> train={len(train)} val={len(val)} test={len(test)}")

    if args.dry_run:
        print("[dry-run] no files written."); return

    out_root = (args.out / "ACF").resolve()
    if test:
        write_split(test, out_root / "test")
    write_split(val or test, out_root / "val")     # val falls back to test
    write_split(train or test, out_root / "train")  # so Ultralytics always resolves
    (out_root / "data.yaml").write_text(
        "# US Real-time gun detection in CCTV (Mock Attack) in YOLO format.\n"
        "# Public; cite Gonzalez et al., Neural Networks 132 (2020) 297-308.\n"
        "# Handgun/Knife only (rifle dropped).\n"
        f"path: {out_root.as_posix()}\n"
        "train: train/images\nval:   val/images\ntest:  test/images\n\n"
        "nc: 2\nnames:\n  0: Handgun\n  1: Knife\n"
    )
    print(f"[OK] wrote {out_root}/data.yaml")
    print("[next] train all backbones then evaluate:")
    print("       PYTHONPATH=src python finish_run.py")


if __name__ == "__main__":
    main()
