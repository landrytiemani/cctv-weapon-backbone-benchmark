#!/usr/bin/env python3
"""
plots.py - Publication figures from the evaluation CSV(s).

Produces, into results_acf/plots/:
  1. trade-off scatter: AP50 vs Params and AP50 vs GFLOPs (the headline efficiency
     figure - shows lightweight ViT backbones matching heavy CNNs at a fraction of
     the cost). CNN baselines vs ViT backbones are colour/marker coded.
  2. grouped bar chart of AP50 and F1 per model.
Saved as both PDF (vector, for the paper) and PNG, at 300 dpi.

Usage (from repo root):
    python plots.py                                   # uses results_acf/acf_comparison.csv
    python plots.py --metric AP50_95
    python plots.py --csv results_acf/acf_comparison.csv --tag acf
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results_acf" / "plots"

# pretty labels + family (cnn vs vit) for colour/marker coding
LABELS = {
    "yolov8n": ("YOLOv8n", "cnn"), "yolov8s": ("YOLOv8s", "cnn"),
    "yolov8m": ("YOLOv8m", "cnn"), "yolov8l": ("YOLOv8l", "cnn"),
    "yolov8x": ("YOLOv8x", "cnn"),
    "efficientvit_yolov8": ("EfficientViT-B1", "vit"),
    "mobilevit_yolov8": ("MobileViT-S", "vit"),
    "efficientformer_yolov8": ("EfficientFormerV2-S2", "vit"),
}
COLORS = {"cnn": "#1f77b4", "vit": "#d62728"}
MARKERS = {"cnn": "o", "vit": "^"}


def load(csv_path: Path):
    rows = []
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            m = r["model"]
            label, fam = LABELS.get(m, (m, "cnn"))
            def fnum(k):
                try:
                    return float(r[k])
                except (KeyError, ValueError, TypeError):
                    return None
            rows.append(dict(model=m, label=label, fam=fam,
                             AP50=fnum("AP50"), AP50_95=fnum("AP50_95"),
                             F1=fnum("F1"), Params=fnum("Params_M"), GFLOPs=fnum("GFLOPs")))
    return rows


def scatter(ax, rows, xkey, ymetric, xlabel):
    for fam in ("cnn", "vit"):
        pts = [r for r in rows if r["fam"] == fam and r[xkey] is not None and r[ymetric] is not None]
        if pts:
            ax.scatter([p[xkey] for p in pts], [p[ymetric] for p in pts],
                       c=COLORS[fam], marker=MARKERS[fam], s=70,
                       label=("YOLOv8 (CNN)" if fam == "cnn" else "Edge-ViT backbone"),
                       edgecolors="black", linewidths=0.5, zorder=3)
    for r in rows:
        if r[xkey] is not None and r[ymetric] is not None:
            ax.annotate(r["label"], (r[xkey], r[ymetric]), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ymetric.replace("_", ":"))
    ax.grid(True, ls=":", lw=0.5, alpha=0.6)
    ax.legend(fontsize=8, loc="lower right")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "results_acf" / "acf_comparison.csv"))
    ap.add_argument("--tag", default=None, help="filename prefix (default from csv name)")
    ap.add_argument("--metric", default="AP50", choices=["AP50", "AP50_95", "F1"],
                    help="y-axis metric for the trade-off figure")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found. Run evaluate.py first.")
    tag = args.tag or csv_path.stem.replace("_comparison", "")
    rows = load(csv_path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Figure 1: accuracy vs efficiency trade-off (params & GFLOPs) ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    scatter(axes[0], rows, "Params", args.metric, "Parameters (M)")
    scatter(axes[1], rows, "GFLOPs", args.metric, "GFLOPs")
    fig.suptitle(f"Accuracy vs. efficiency ({tag})", fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{tag}_tradeoff.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: per-model AP50 + F1 bars ---
    rows_sorted = sorted(rows, key=lambda r: (r["AP50"] is None, -(r["AP50"] or 0)))
    labels = [r["label"] for r in rows_sorted]
    ap50 = [r["AP50"] or 0 for r in rows_sorted]
    f1 = [r["F1"] or 0 for r in rows_sorted]
    x = range(len(labels)); w = 0.4
    fig2, ax = plt.subplots(figsize=(max(6, len(labels) * 1.1), 4))
    ax.bar([i - w / 2 for i in x], ap50, w, label="AP50", color="#1f77b4")
    ax.bar([i + w / 2 for i in x], f1, w, label="F1", color="#ff7f0e")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("score (%)"); ax.set_title(f"Per-model AP50 / F1 ({tag})", fontsize=11)
    ax.grid(True, axis="y", ls=":", lw=0.5, alpha=0.6); ax.legend(fontsize=8)
    fig2.tight_layout()
    for ext in ("pdf", "png"):
        fig2.savefig(OUT_DIR / f"{tag}_bars.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig2)

    print(f"[OK] wrote {OUT_DIR}/{tag}_tradeoff.(pdf|png) and {tag}_bars.(pdf|png)")


if __name__ == "__main__":
    main()
