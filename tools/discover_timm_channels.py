"""
Print the per-stage output channel counts for a timm backbone, so the
``mobilevit_yolov8.yaml`` / ``efficientformer_yolov8.yaml`` configs use
the right Concat / C2f channel widths.

Usage:
    python scripts/discover_timm_channels.py mobilevit_s
    python scripts/discover_timm_channels.py efficientformer_l1
"""

import sys
import timm


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/discover_timm_channels.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    # Ask timm for ALL stages first; out_indices count varies by model.
    model = timm.create_model(
        model_name, pretrained=False, features_only=True,
    )
    info = model.feature_info
    chans = info.channels()
    reds = info.reduction()
    print(f"\n{model_name}: {len(chans)} feature stages")
    for i, (ch, red) in enumerate(zip(chans, reds)):
        print(f"  stage_idx={i}  stride=/{red}  channels={ch}")
    print()


if __name__ == "__main__":
    main()
