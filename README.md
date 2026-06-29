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

Populate from `results_acf/acf_comparison.csv` after running `evaluate.py`:

| Model | AP50 | AP50:95 | P | R | F1 | Params (M) | GFLOPs |
|---|---|---|---|---|---|---|---|
| _to be filled_ | | | | | | | |

Edge latency (e.g. Jetson-Nano FPS) is hardware-dependent and reported separately
from your own device measurements.

## Known limitations

- **Knife is rare in CCTV.** The US-CCTV set is gun-centric; knife instances are
  few. The benchmark is therefore **handgun-dominated**, and the Knife column
  should be read as indicative only (small support), not as a robust per-class
  result. Treat the headline numbers as a handgun-detection benchmark.
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
