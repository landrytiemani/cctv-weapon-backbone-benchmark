"""
EfficientViT-YOLOv8: Hybrid model combining EfficientViT backbone with
YOLOv8 PANet neck and detection head.

This module creates a complete detection model by:
1. Using EfficientViT-B1 as the backbone (CGA-based, O(n) attention)
2. Attaching YOLOv8's PANet neck for multi-scale feature aggregation
3. Using YOLOv8's detection head for bounding box + class prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv, C2f, SPPF, Detect, Concat
from models.efficientvit_backbone import EfficientViTBackbone


class EfficientViTYOLOv8(nn.Module):
    """Complete EfficientViT-YOLOv8 detection model.

    Backbone: EfficientViT-B1 (outputs P3=64ch, P4=128ch, P5=256ch)
    Neck: YOLOv8-style PANet (top-down + bottom-up feature aggregation)
    Head: YOLOv8 Detect head (3-scale detection)
    """

    def __init__(self, nc=2, ch=3):
        super().__init__()
        self.nc = nc
        self.names = {0: "Handgun", 1: "Knife"}

        # === Backbone ===
        self.backbone = EfficientViTBackbone(in_channels=ch)
        # P3=64, P4=128, P5=256

        # === Neck (PANet - top-down path) ===
        # Process P5 and upsample to concat with P4
        self.lateral1 = Conv(256, 128, 1)  # reduce P5 channels
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2f_td1 = C2f(256, 128, n=1)  # concat(P4, up(lat(P5))): 128+128=256 -> 128

        # Process and upsample to concat with P3
        self.lateral2 = Conv(128, 64, 1)  # reduce channels
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2f_td2 = C2f(128, 64, n=1)  # concat(P3, up(lat(...))): 64+64=128 -> 64

        # === Neck (PANet - bottom-up path) ===
        self.down1 = Conv(64, 64, 3, 2)  # stride 2 downsample
        self.c2f_bu1 = C2f(192, 128, n=1)  # concat(down(td2), td1): 64+128=192 -> 128

        self.down2 = Conv(128, 128, 3, 2)  # stride 2 downsample
        self.c2f_bu2 = C2f(384, 256, n=1)  # concat(down(bu1), P5): 128+256=384 -> 256

        # === Detection Head ===
        self.detect = Detect(nc=nc, ch=(64, 128, 256))

        # Store stride info
        self.stride = torch.tensor([8., 16., 32.])
        self.detect.stride = self.stride

        # Store as nn.Sequential so model.model[-1] works for v8DetectionLoss
        self.model = nn.Sequential(self.detect)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Backbone
        p3, p4, p5 = self.backbone(x)

        # Top-down path
        td1 = self.up1(self.lateral1(p5))
        td1 = self.c2f_td1(torch.cat([p4, td1], dim=1))  # 128 ch at P4 scale

        td2 = self.up2(self.lateral2(td1))
        td2 = self.c2f_td2(torch.cat([p3, td2], dim=1))  # 64 ch at P3 scale

        # Bottom-up path
        bu1 = self.c2f_bu1(torch.cat([self.down1(td2), td1], dim=1))  # 128 ch at P4 scale
        bu2 = self.c2f_bu2(torch.cat([self.down2(bu1), p5], dim=1))   # 256 ch at P5 scale

        # Detection head
        return self.detect([td2, bu1, bu2])


# For standalone testing
if __name__ == "__main__":
    model = EfficientViTYOLOv8(nc=2)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        out = model(x)

    # In eval mode, Detect returns (y, preds) where y is the inference output
    if isinstance(out, tuple):
        print(f"Inference output: {out[0].shape}")
        print(f"Training output keys: {out[1].keys() if isinstance(out[1], dict) else type(out[1])}")
    else:
        print(f"Output: {out.shape if hasattr(out, 'shape') else type(out)}")

    try:
        from thop import profile
        flops, _ = profile(model.backbone, inputs=(x,), verbose=False)
        print(f"Backbone GFLOPs: {flops/1e9:.2f}")
    except Exception as e:
        print(f"FLOPs computation: {e}")
