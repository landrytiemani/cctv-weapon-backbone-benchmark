# Edge-AI backbone benchmark for CCTV weapon detection

A **self-contained** benchmark of edge-oriented backbones for handheld-weapon
(handgun + knife) detection on **real surveillance (CCTV) footage**, built
entirely on a **public** dataset so the results are reproducible. Two edge
Vision-Transformer backbones — **EfficientViT-B1** (linear attention) and
**MobileViT-S** (hybrid) — are integrated into the **YOLOv8** detection framework
and compared against the five YOLOv8 variants (n/s/m/l/x). The PANet neck and the
YOLOv8 detection head are kept identical across all detectors, so the **backbone
is the only varying component**.

This repository has **no external project dependencies** beyond the Python
packages in `requirements.txt`; everything needed (model code, configs, training
recipe, evaluation, figures) is included.

## Dataset

**US Real-time gun detection in CCTV — "Mock Attack" subset (ACF)**,
Deepknowledge-US / University of Seville. Real CCTV frames of staged attacks;
the only weapon classes kept here are **handgun** and **knife** (rifle/long-gun
annotations are dropped, see Method notes).

- License: **CC BY-NC 4.0** — research / non-commercial use, with attribution.
- Source: https://github.com/Deepknowledge-US/US-Real-time-gun-detection-in-CCTV-An-open-problem-dataset
- Cite: González, J.L.S., Zaccaro, C., Álvarez-García, J.A., Soria-Morillo, L.M.,
  Sancho-Caparrini, F. (2020). *Real-time gun detection in CCTV: An open problem.*
  Neural Networks 132, 297–308. https://doi.org/10.1016/j.neunet.2020.09.013

The dataset itself is **not** redistributed here (and its non-commercial licence
forbids commercial redistribution); `prepare_acf.py` converts your own local
download into the YOLO format this repo expects.

## Install

```bash
git clone https://github.com/landrytiemani/cctv-weapon-backbone-benchmark.git
cd cctv-weapon-backbone-benchmark
pip install -r requirements.txt
```

A CUDA GPU is required for training.

## Usage

```bash
# 1. Download the US-CCTV "Mock Attack" images + VOC annotations from the repo
#    above, then convert to a leakage-safe YOLO split (grouped by camera/segment):
python prepare_acf.py --src /path/to/cctv_mock_attack --out ./data --val-frac 0.15
#    -> data/ACF/{train,val,test}/{images,labels} + data/ACF/data.yaml

# 2. Train all seven backbones (one shared recipe), or one at a time:
PYTHONPATH=src python train.py --model all
PYTHONPATH=src python train.py --model yolov8x
PYTHONPATH=src python train.py --model efficientvit
PYTHONPATH=src python train.py --model mobilevit
#    resume-safe orchestrator (skips finished runs, resumes interrupted ones):
PYTHONPATH=src python finish_run.py
#    quick pipeline check before the real runs:
PYTHONPATH=src python train.py --model yolov8n --smoke

# 3. Evaluate on the test split -> results_acf/acf_comparison.{csv,json}:
PYTHONPATH=src python evaluate.py

# 4. Publication figures and qualitative panels:
PYTHONPATH=src python plots.py
PYTHONPATH=src python qualitative.py --n 50 --conf 0.15
```

## Repository layout

```
.
├── prepare_acf.py       # US-CCTV (VOC XML) -> YOLO format + leakage-safe split
├── train.py             # 7-backbone trainer, one shared SGD recipe
├── finish_run.py        # resume-safe orchestrator (train -> evaluate)
├── evaluate.py          # AP50 / AP50:95 / P / R / F1 / Params / GFLOPs table
├── plots.py             # accuracy-vs-efficiency trade-off + per-model bars
├── qualitative.py       # GT-vs-all-models panels with a weapon-region zoom inset
├── configs/             # YOLOv8 model configs (EfficientViT, MobileViT, ...)
├── src/models/          # EfficientViT + timm backbone integration, pretrained init
├── tools/               # discover_timm_channels.py (backbone channel helper)
├── data/ACF/            # generated dataset (gitignored)
├── results_acf/         # training runs + comparison tables/figures (gitignored)
├── requirements.txt
└── LICENSE              # MIT (code)
```

## Method notes

- **Classes:** handgun → `Handgun (0)`, knife → `Knife (1)`.
- **Rifles/long guns dropped:** rifle/shotgun annotations are removed, but the
  frame is **kept**, so a spurious detection on a rifle counts as a false
  positive — the conservative choice for a handgun/knife detector.
- **Leakage-safe split:** the train/val/test partition is grouped by
  camera/segment (`<id>_frame_<n>`), so frames from one clip never straddle
  splits. `--random-split` opts into a per-frame split (leaks; not recommended).
- **Distractor negatives:** frames with no kept weapon are retained as negatives
  (empty labels) by default (`--keep-negatives`), which improves precision.
- **Recipe (identical for every detector):** SGD, lr0=0.05, momentum=0.9,
  weight_decay=5e-4, batch=32, 300 epochs, linear LR decay (lrf=0.01),
  patience=100, imgsz=640, augment = mosaic + HSV + hflip + translate + scale
  (no mixup/copy_paste). ViT backbones use a 10-epoch warm-up to protect the
  pretrained transformer while the randomly-initialised neck/head settle.

## Results

CCTV (ACF) test split, two classes (Handgun + Knife), `imgsz=640`. All seven
detectors were trained and evaluated on a single **NVIDIA A100** GPU under the
identical recipe above. `AP50_H`/`AP50_K` are per-class AP50 (Handgun/Knife);
`FPS` is single-image, end-to-end throughput at batch 1 on the A100. Sorted by
mean AP50 (from `results_acf/acf_comparison.csv`):

| Model | AP50_H | AP50_K | AP50 | AP50:95 | F1 | P | R | FPS | Params (M) | GFLOPs |
|---|---|---|---|---|---|---|---|---|---|---|
| **MobileViT-S (ours)** | **48.56** | **65.60** | **57.08** | **28.26** | **62.28** | 78.62 | **51.56** | 73.6 | 7.01 | 31.16 |
| YOLOv8x | 47.35 | 61.12 | 54.23 | 20.82 | 58.00 | 86.02 | 43.75 | 94.7 | 61.60 | 226.70 |
| YOLOv8n | 45.52 | 59.77 | 52.64 | 20.51 | 59.05 | **89.03** | 44.18 | **138.9** | **2.68** | **6.82** |
| YOLOv8s | 42.80 | 62.07 | 52.44 | 22.55 | 55.85 | 87.47 | 41.02 | 137.5 | 9.83 | 23.35 |
| EfficientViT-B1 (ours) | 44.67 | 49.25 | 46.96 | 20.43 | 56.38 | 82.27 | 42.88 | 75.0 | 11.21 | 22.81 |
| YOLOv8m | 42.48 | 50.57 | 46.52 | 18.10 | 52.95 | 65.49 | 44.44 | 112.6 | 23.20 | 67.43 |
| YOLOv8l | 44.65 | 44.98 | 44.82 | 20.79 | 58.20 | 81.90 | 45.14 | 98.2 | 39.43 | 145.19 |

The **MobileViT-S** backbone gives the best mean AP50, AP50:95, F1 **and the best
per-class AP50 on both classes** while using **~9× fewer parameters** and **~7×
fewer FLOPs** than the strongest CNN baseline (YOLOv8x) — the most favourable
accuracy/efficiency trade-off here. Note the **FPS/GFLOPs mismatch**: the ViT
backbones have the fewest FLOPs but the lowest wall-clock FPS (their attention /
patch ops are memory-bandwidth-bound and unfused under PyTorch eager mode);
TensorRT/ONNX fusion is expected to narrow this gap. On-device (Jetson) latency is
left to deployment.

## Known limitations

- **Knife has fewer instances than Handgun** (the US-CCTV set is gun-centric), but
  it is in fact the **better-detected** class — knife AP50 exceeds handgun AP50 for
  every backbone. Its per-class numbers are reported with that smaller support in
  mind, not treated as unreliable.
- **EfficientFormerV2** is *not* in the default pipeline: its resolution-coupled
  attention bias is incompatible with 640×640 detection in current `timm`
  (token-count vs attention-bias mismatch). It is documented as
  attempted-but-incompatible rather than benchmarked.

## License

Code: MIT (see `LICENSE`). Dataset: **CC BY-NC 4.0** (González et al., 2020) —
non-commercial use only; attribute the original authors and do not redistribute
the imagery.

## Acknowledgements

Backbone implementations build on the EfficientViT and MobileViT works and the
`timm` library; the YOLOv8 framework is provided by Ultralytics.
