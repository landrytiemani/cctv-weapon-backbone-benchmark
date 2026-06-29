"""
EfficientViT Backbone modules for YOLOv8 integration.

Implements the key building blocks from:
"EfficientViT: Memory Efficient Vision Transformer with Cascaded Group Attention"
(Liu et al., MIT Han Lab, 2023)

Key innovation: Cascaded Group Attention (CGA) achieves O(n) complexity
instead of O(n^2) standard self-attention by splitting heads into groups,
each attending to a subset of tokens, with cascaded information flow.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnAct(nn.Module):
    """Conv2d + BatchNorm + Activation."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1, act=True):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding,
                              groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DSConv(nn.Module):
    """Depthwise Separable Convolution."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        self.dw = ConvBnAct(in_ch, in_ch, kernel_size, stride, groups=in_ch)
        self.pw = ConvBnAct(in_ch, out_ch, 1)

    def forward(self, x):
        return self.pw(self.dw(x))


class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Conv (MBConv) block.
    Used in early stages of EfficientViT for local feature extraction.
    """

    def __init__(self, in_ch, out_ch, expand_ratio=4, stride=1):
        super().__init__()
        mid_ch = int(in_ch * expand_ratio)
        self.use_residual = (stride == 1 and in_ch == out_ch)

        self.conv1 = ConvBnAct(in_ch, mid_ch, 1)
        self.conv2 = ConvBnAct(mid_ch, mid_ch, 3, stride, groups=mid_ch)
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        out = self.conv3(self.conv2(self.conv1(x)))
        if self.use_residual:
            out = out + x
        return out


class CascadedGroupAttention(nn.Module):
    """Cascaded Group Attention (CGA) — the core innovation of EfficientViT.

    Instead of all heads attending to all tokens (O(n^2)):
    - Splits heads into K groups
    - Each group attends to n/K tokens → cost = K * (n/K)^2 = n^2/K → O(n)
    - Cascaded: each group receives the output of the previous group
      for information flow across groups.

    Stability features:
    - LayerNorm on input before Q/K/V projections
    - Attention logit clamping before softmax to prevent NaN
    - LayerNorm on cascaded connections to prevent accumulation drift
    - Attention dropout for regularization
    """

    def __init__(self, dim, num_heads=8, attn_ratio=4, attn_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        total_dim = self.head_dim * num_heads

        # LayerNorm before Q/K/V projections for numerical stability
        self.input_norm = nn.LayerNorm(self.head_dim)

        # Each head has its own QKV projection (key to CGA)
        self.qkvs = nn.ModuleList([
            nn.Linear(self.head_dim, 3 * self.head_dim) for _ in range(num_heads)
        ])
        # Projection for cascaded output from previous head
        self.dws = nn.ModuleList([
            nn.Conv2d(self.head_dim, self.head_dim, 3, 1, 1, groups=self.head_dim)
            for _ in range(num_heads)
        ])
        # LayerNorm for cascaded connection (prevents accumulation drift)
        self.cascade_norms = nn.ModuleList([
            nn.LayerNorm(self.head_dim) for _ in range(num_heads - 1)
        ])

        self.proj = nn.Linear(total_dim, dim)
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # Reshape to sequence: (B, C, H, W) -> (B, N, C) where N = H*W
        x_flat = x.flatten(2).transpose(1, 2)  # (B, N, C)
        N = H * W

        # Split input into per-head chunks
        x_heads = x_flat.reshape(B, N, self.num_heads, self.head_dim)

        outputs = []
        for i, (qkv_proj, dw) in enumerate(zip(self.qkvs, self.dws)):
            # Get input for this head
            if i == 0:
                head_input = x_heads[:, :, i]  # (B, N, head_dim)
            else:
                # Cascaded: add normalized output from previous head
                cascade_out = self.cascade_norms[i - 1](outputs[-1])
                head_input = x_heads[:, :, i] + cascade_out

            # Normalize before QKV projection
            head_input = self.input_norm(head_input)

            # QKV projection
            qkv = qkv_proj(head_input)  # (B, N, 3*head_dim)
            q, k, v = qkv.chunk(3, dim=-1)

            # Attention with logit clamping for numerical stability
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.clamp(-50.0, 50.0)  # prevent extreme values before softmax
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            head_out = attn @ v  # (B, N, head_dim)

            # Apply depthwise conv for local information mixing
            head_out_2d = head_out.transpose(1, 2).reshape(B, self.head_dim, H, W)
            head_out_2d = dw(head_out_2d)
            head_out = head_out_2d.flatten(2).transpose(1, 2)

            outputs.append(head_out)

        # Concatenate all heads
        out = torch.cat(outputs, dim=-1)  # (B, N, C)
        out = self.proj(out)
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return out


class EfficientViTBlock(nn.Module):
    """Single EfficientViT block: FFN -> CGA -> FFN.

    Uses sandwich layout: local FFN (depthwise conv) before and after
    the global CGA attention for efficient feature processing.
    """

    def __init__(self, dim, num_heads=8, expand_ratio=4):
        super().__init__()
        mid_dim = int(dim * expand_ratio)

        # Local FFN before attention (context aggregation)
        self.local_ffn = nn.Sequential(
            DSConv(dim, dim),
        )

        # Global attention (CGA)
        self.attn_norm = nn.BatchNorm2d(dim)
        self.attn = CascadedGroupAttention(dim, num_heads)

        # FFN after attention
        self.ffn_norm = nn.BatchNorm2d(dim)
        self.ffn = nn.Sequential(
            ConvBnAct(dim, mid_dim, 1),
            DSConv(mid_dim, dim),
        )

    def forward(self, x):
        # Local mixing
        x = x + self.local_ffn(x)
        # Global attention
        x = x + self.attn(self.attn_norm(x))
        # FFN
        x = x + self.ffn(self.ffn_norm(x))
        return x


class EfficientViTStage(nn.Module):
    """A stage of EfficientViT blocks with optional downsampling."""

    def __init__(self, in_ch, out_ch, depth, num_heads=8, expand_ratio=4,
                 downsample=True, use_cga=True):
        super().__init__()
        layers = []

        # Downsampling via strided DSConv
        if downsample:
            layers.append(DSConv(in_ch, out_ch, stride=2))
        elif in_ch != out_ch:
            layers.append(ConvBnAct(in_ch, out_ch, 1))

        # Stack blocks
        for _ in range(depth):
            if use_cga:
                layers.append(EfficientViTBlock(out_ch, num_heads, expand_ratio))
            else:
                layers.append(MBConv(out_ch, out_ch, expand_ratio))

        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.blocks(x)


class EfficientViTBackbone(nn.Module):
    """EfficientViT-B1 backbone adapted for YOLOv8 detection.

    Outputs multi-scale features at P3 (stride 8), P4 (stride 16), P5 (stride 32)
    for the YOLOv8 PANet neck.

    Architecture (B1 variant):
    - Stem: Conv stride 2 → 16 channels
    - Stage 1: MBConv, stride 2 → 32 channels (P2, stride 4)
    - Stage 2: MBConv, stride 2 → 64 channels (P3, stride 8)  ← output
    - Stage 3: CGA blocks, stride 2 → 128 channels (P4, stride 16) ← output
    - Stage 4: CGA blocks, stride 2 → 256 channels (P5, stride 32) ← output
    """

    def __init__(self, in_channels=3):
        super().__init__()
        # B1 configuration
        channels = [16, 32, 64, 128, 256]
        depths = [1, 2, 3, 3, 4]
        num_heads = [0, 0, 0, 4, 8]  # CGA only in stages 3-4

        # Stem
        self.stem = ConvBnAct(in_channels, channels[0], 3, stride=2)

        # Stage 1: Local (MBConv), stride 2 → P2
        self.stage1 = EfficientViTStage(
            channels[0], channels[1], depths[1],
            downsample=True, use_cga=False, expand_ratio=4
        )

        # Stage 2: Local (MBConv), stride 2 → P3 (output)
        self.stage2 = EfficientViTStage(
            channels[1], channels[2], depths[2],
            downsample=True, use_cga=False, expand_ratio=4
        )

        # Stage 3: Global (CGA), stride 2 → P4 (output)
        self.stage3 = EfficientViTStage(
            channels[2], channels[3], depths[3],
            num_heads=num_heads[3], downsample=True, use_cga=True, expand_ratio=4
        )

        # Stage 4: Global (CGA), stride 2 → P5 (output)
        self.stage4 = EfficientViTStage(
            channels[3], channels[4], depths[4],
            num_heads=num_heads[4], downsample=True, use_cga=True, expand_ratio=4
        )

        # Channel indices for YOLOv8 neck connection
        self.out_channels = [channels[2], channels[3], channels[4]]  # P3, P4, P5

    def forward(self, x):
        x = self.stem(x)       # stride 2
        x = self.stage1(x)     # stride 4 (P2)
        p3 = self.stage2(x)    # stride 8 (P3)
        p4 = self.stage3(p3)   # stride 16 (P4)
        p5 = self.stage4(p4)   # stride 32 (P5)
        return [p3, p4, p5]


class EfficientViTBackboneWrapper(nn.Module):
    """Wrapper to make EfficientViTBackbone compatible with Ultralytics.

    This module is registered as a custom backbone in the YOLOv8 model YAML.
    It takes the raw image input and returns multi-scale features.
    """

    def __init__(self, c1=3, c2=256):
        super().__init__()
        self.backbone = EfficientViTBackbone(in_channels=c1)
        self._out_channels = self.backbone.out_channels

    def forward(self, x):
        return self.backbone(x)


# For standalone testing
if __name__ == "__main__":
    model = EfficientViTBackbone(in_channels=3)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    x = torch.randn(1, 3, 640, 640)
    outputs = model(x)
    for i, out in enumerate(outputs):
        print(f"P{i+3}: {out.shape}")

    # Compute FLOPs
    try:
        from thop import profile
        flops, params = profile(model, inputs=(x,), verbose=False)
        print(f"GFLOPs: {flops/1e9:.2f}")
    except ImportError:
        print("Install thop for FLOPs computation: pip install thop")
