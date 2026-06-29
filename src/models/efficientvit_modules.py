"""
MIT Han Lab EfficientViT-B1 modules for Ultralytics YAML integration.

Implements the architecture from:
"EfficientViT: Lightweight Multi-Scale Attention for High-Resolution Dense Prediction"
(Cai et al., MIT Han Lab, ICCV 2023)

Key innovation: LiteMLA (Lightweight Multi-Scale Linear Attention) — O(n) complexity
via ReLU-based linear attention with multi-scale token aggregation.

B1 config: width=[16,32,64,128,256], depth=[1,2,3,3,4], dim=16
ImageNet-1K Top-1: 79.39% (pretrained weights available)

YAML-compatible modules follow Ultralytics convention: __init__(self, c1, c2, *args)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Core building blocks (matching official MIT implementation)
# ============================================================

class ConvLayer(nn.Module):
    """Conv2d + optional BatchNorm + optional Activation.

    Matches official EfficientViT ConvLayer with support for
    bias/norm/act control via tuples.
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1,
                 use_bias=False, norm=True, act=True):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding,
                              groups=groups, bias=use_bias)
        self.norm = nn.BatchNorm2d(out_ch) if norm else nn.Identity()
        self.act = nn.Hardswish(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class DSConv(nn.Module):
    """Depthwise Separable Convolution (depth_conv + point_conv)."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        self.depth_conv = ConvLayer(in_ch, in_ch, kernel_size, stride, groups=in_ch)
        self.point_conv = ConvLayer(in_ch, out_ch, 1)

    def forward(self, x):
        return self.point_conv(self.depth_conv(x))


class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Conv block.

    Structure: pointwise expand -> depthwise -> pointwise project
    Optional residual connection when stride=1 and in_ch==out_ch.
    """

    def __init__(self, in_ch, out_ch, expand_ratio=4, stride=1, fewer_norm=False):
        super().__init__()
        mid_ch = int(in_ch * expand_ratio)
        self.use_residual = (stride == 1 and in_ch == out_ch)

        if fewer_norm:
            # Official EfficientViT uses fewer norms in attention stages
            self.inverted_conv = nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, 1, bias=True),
                nn.Hardswish(inplace=True),
            )
            self.depth_conv = nn.Sequential(
                nn.Conv2d(mid_ch, mid_ch, 3, stride, 1, groups=mid_ch, bias=True),
                nn.Hardswish(inplace=True),
            )
            self.point_conv = nn.Sequential(
                nn.Conv2d(mid_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.inverted_conv = ConvLayer(in_ch, mid_ch, 1)
            self.depth_conv = ConvLayer(mid_ch, mid_ch, 3, stride, groups=mid_ch)
            self.point_conv = nn.Sequential(
                nn.Conv2d(mid_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.point_conv(self.depth_conv(self.inverted_conv(x)))
        if self.use_residual:
            out = out + x
        return out


class LiteMLA(nn.Module):
    """Lightweight Multi-Scale Linear Attention.

    Core attention mechanism of MIT EfficientViT:
    - Uses ReLU kernel for linear attention (O(n) instead of O(n^2))
    - Multi-scale feature aggregation via depthwise convolutions
    - No softmax — uses ReLU(Q) * (ReLU(K)^T * V) normalization

    Args:
        dim: input/output channels
        heads: number of attention heads (default: dim // 16 for B1)
        head_dim: dimension per head (default: 16 for B1)
        scales: kernel sizes for multi-scale aggregation (default: (5,))
        eps: numerical stability epsilon for normalization
    """

    def __init__(self, dim, heads=None, head_dim=16, scales=(5,), eps=1e-15):
        super().__init__()
        if heads is None:
            heads = dim // head_dim
        self.heads = heads
        self.head_dim = head_dim
        self.total_dim = heads * head_dim
        self.num_scales = len(scales)
        self.eps = eps

        # QKV projection (1x1 conv)
        self.qkv = ConvLayer(dim, 3 * self.total_dim, 1, use_bias=False, norm=False, act=False)

        # Multi-scale aggregation branches
        self.aggreg = nn.ModuleList()
        for scale in scales:
            self.aggreg.append(nn.Sequential(
                nn.Conv2d(3 * self.total_dim, 3 * self.total_dim, scale,
                          padding=scale // 2, groups=3 * self.total_dim, bias=False),
                nn.Conv2d(3 * self.total_dim, 3 * self.total_dim, 1,
                          groups=3 * self.heads, bias=False),
            ))

        # Output projection
        self.proj = ConvLayer(self.total_dim * (1 + self.num_scales), dim, 1,
                              use_bias=False, norm=True, act=False)

    def forward(self, x):
        B, C, H, W = x.shape

        # QKV projection
        qkv = self.qkv(x)  # (B, 3*total_dim, H, W)

        # Multi-scale aggregation: concat original + each scale's output
        multi_scale_qkv = [qkv]
        for aggreg in self.aggreg:
            multi_scale_qkv.append(aggreg(qkv))

        # Stack and reshape for attention
        # Each qkv: (B, 3*total_dim, H, W) → total (1+num_scales) of them
        multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)  # (B, 3*total_dim*(1+S), H, W)

        # Reshape: (B, heads*(1+S), 3, head_dim, H*W)
        num_heads_total = self.heads * (1 + self.num_scales)
        multi_scale_qkv = multi_scale_qkv.reshape(B, num_heads_total, 3 * self.head_dim, H * W)

        # Split into Q, K, V
        q, k, v = multi_scale_qkv.split(self.head_dim, dim=2)
        # q, k, v: (B, num_heads_total, head_dim, N) where N = H*W

        # Force float32 for attention math — V@K^T produces large intermediates
        # that overflow float16 under AMP, causing NaN
        input_dtype = q.dtype
        q = F.relu(q).float()
        k = F.relu(k).float()
        v = v.float()

        # Linear attention: O(n) complexity
        # out = (V @ K^T) @ Q / (1^T @ K^T @ Q)
        v_pad = F.pad(v, (0, 0, 0, 1), value=1.0)  # (B, heads, head_dim+1, N)
        vk = v_pad @ k.transpose(-2, -1)  # (B, heads, head_dim+1, head_dim)
        out = vk @ q  # (B, heads, head_dim+1, N)

        # Normalize and cast back
        out = out[:, :, :-1] / (out[:, :, -1:].clamp(min=self.eps))
        out = out.to(input_dtype)  # back to original dtype (fp16 under AMP)

        # Reshape back to spatial
        out = out.reshape(B, self.total_dim * (1 + self.num_scales), H, W)

        # Project
        out = self.proj(out)
        return out


class EfficientViTBlock(nn.Module):
    """EfficientViT block: LiteMLA attention + MBConv local processing.

    Each block has two residual sub-blocks:
    1. context_module: LiteMLA for global context (linear attention)
    2. local_module: MBConv for local feature refinement
    """

    def __init__(self, dim, head_dim=16, expand_ratio=4, scales=(5,)):
        super().__init__()
        # Global context via linear attention
        self.context_module = LiteMLA(dim, head_dim=head_dim, scales=scales)
        # Local refinement via MBConv
        self.local_module = MBConv(dim, dim, expand_ratio, fewer_norm=True)

    def forward(self, x):
        x = x + self.context_module(x)  # global attention (residual)
        x = x + self.local_module(x)     # local MBConv (residual)
        return x


# ============================================================
# YAML-compatible stage modules
# ============================================================

class EfficientViTStem(nn.Module):
    """Input stem: Conv + residual DSConv blocks.

    YAML-compatible: (c1, c2, depth).
    Applies stride-2 conv, then 'depth' residual DSConv blocks.
    Matches official input_stem structure.

    Args:
        c1: input channels (3 for RGB)
        c2: output channels (16 for B1)
        depth: number of residual DSConv blocks (1 for B1)
    """

    def __init__(self, c1, c2, depth=1):
        super().__init__()
        layers = [ConvLayer(c1, c2, 3, stride=2)]
        for _ in range(depth):
            layers.append(DSConv(c2, c2))  # residual is handled manually
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x = self.layers[0](x)  # stem conv
        for layer in self.layers[1:]:
            x = x + layer(x)  # residual DSConv blocks
        return x


class EfficientViTLocalStage(nn.Module):
    """Local feature extraction stage using MBConv blocks.

    YAML-compatible: (c1, c2, depth, expand_ratio).
    First MBConv downsamples 2x (stride=2), remaining are stride=1 with residual.

    Args:
        c1: input channels
        c2: output channels
        depth: number of MBConv blocks (including the downsample block)
        expand_ratio: MBConv expansion ratio (4 for B1)
    """

    def __init__(self, c1, c2, depth=1, expand_ratio=4):
        super().__init__()
        blocks = []
        # First block: downsample (stride=2, no residual)
        blocks.append(MBConv(c1, c2, expand_ratio, stride=2))
        # Remaining blocks: stride=1 with residual
        for _ in range(depth - 1):
            blocks.append(MBConv(c2, c2, expand_ratio))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        x = self.blocks[0](x)  # downsample (no residual, handled inside MBConv)
        for block in self.blocks[1:]:
            x = block(x)  # residual handled inside MBConv
        return x


class EfficientViTAttentionStage(nn.Module):
    """Attention stage using LiteMLA + MBConv blocks.

    YAML-compatible: (c1, c2, depth, head_dim, expand_ratio).
    First MBConv downsamples 2x, then 'depth' EfficientViTBlocks with LiteMLA.

    Args:
        c1: input channels
        c2: output channels
        depth: number of EfficientViT attention blocks
        head_dim: dimension per attention head (16 for B1)
        expand_ratio: MBConv expansion ratio (4 for B1)
    """

    def __init__(self, c1, c2, depth=1, head_dim=16, expand_ratio=4):
        super().__init__()
        # Downsample MBConv (fewer_norm=True to match official)
        self.downsample = MBConv(c1, c2, expand_ratio, stride=2, fewer_norm=True)
        # Attention blocks
        self.blocks = nn.ModuleList([
            EfficientViTBlock(c2, head_dim, expand_ratio, scales=(5,))
            for _ in range(depth)
        ])

    def forward(self, x):
        x = self.downsample(x)
        for block in self.blocks:
            x = block(x)
        return x


# ============================================================
# Registration with Ultralytics
# ============================================================

def register_efficientvit_modules():
    """Register EfficientViT modules with Ultralytics so YAML configs can use them.

    Adds module classes to ultralytics.nn.tasks namespace and patches parse_model
    to include them in base_modules for proper channel tracking.
    """
    import inspect
    import pathlib
    import re
    import ultralytics.nn.tasks as tasks

    # Step 1: Add to namespace so globals()[module_name] works
    tasks.EfficientViTStem = EfficientViTStem
    tasks.EfficientViTLocalStage = EfficientViTLocalStage
    tasks.EfficientViTAttentionStage = EfficientViTAttentionStage

    if getattr(tasks, "_efficientvit_patched", False):
        print("[EfficientViT] Modules already registered")
        return

    # Step 2: insert our modules into parse_model's base-module membership set so
    # Ultralytics prepends c1 and tracks output channels. Read the PRISTINE source
    # from file (a prior registration may have re-exec'd tasks, misaligning
    # inspect.getsource), and match across Ultralytics versions:
    #   named var:  base_modules = frozenset({...}) / ({...}) / {...}
    #   inline set: if m in { Conv, ... }:           (e.g. 8.3.x)
    try:
        full = pathlib.Path(tasks.__file__).read_text(encoding="utf-8")
        mobj = re.search(r"\ndef parse_model\(.*?(?=\ndef |\Z)", full, re.S)
        src = mobj.group(0) if mobj else inspect.getsource(tasks.parse_model)
    except Exception:
        src = inspect.getsource(tasks.parse_model)

    names = "EfficientViTStem, EfficientViTLocalStage, EfficientViTAttentionStage, "
    patched = None
    for pat in (r"(base_modules\s*=\s*frozenset\(\s*[\{\(\[])",
                r"(base_modules\s*=\s*[\{\(\[])",
                r"(if m in \{)"):
        new = re.sub(pat, r"\1" + names, src, count=1)
        if new != src:
            patched = new
            break
    if patched is None:
        raise RuntimeError(
            "Could not patch Ultralytics parse_model to register EfficientViT "
            "modules. Your installed ultralytics differs from what this adapter "
            "expects. Pin a compatible version (e.g. ultralytics==8.3.0)."
        )

    code = compile(patched, tasks.__file__, "exec")
    exec(code, tasks.__dict__)
    tasks._efficientvit_patched = True
    print("[EfficientViT] Registered MIT EfficientViT-B1 modules with Ultralytics")


# ============================================================
# Standalone testing
# ============================================================

if __name__ == "__main__":
    # Test individual modules
    print("Testing EfficientViT-B1 modules...\n")

    stem = EfficientViTStem(3, 16, depth=1)
    stage1 = EfficientViTLocalStage(16, 32, depth=2, expand_ratio=4)
    stage2 = EfficientViTLocalStage(32, 64, depth=3, expand_ratio=4)
    stage3 = EfficientViTAttentionStage(64, 128, depth=3, head_dim=16, expand_ratio=4)
    stage4 = EfficientViTAttentionStage(128, 256, depth=4, head_dim=16, expand_ratio=4)

    x = torch.randn(1, 3, 640, 640)
    x = stem(x)
    print(f"Stem:   {x.shape}")  # [1, 16, 320, 320]
    x = stage1(x)
    print(f"Stage1: {x.shape}")  # [1, 32, 160, 160]
    x = stage2(x)
    print(f"Stage2: {x.shape}")  # [1, 64, 80, 80]
    p3 = x
    x = stage3(x)
    print(f"Stage3: {x.shape}")  # [1, 128, 40, 40]
    p4 = x
    x = stage4(x)
    print(f"Stage4: {x.shape}")  # [1, 256, 20, 20]
    p5 = x

    total = sum(p.numel() for p in
                list(stem.parameters()) + list(stage1.parameters()) +
                list(stage2.parameters()) + list(stage3.parameters()) +
                list(stage4.parameters()))
    print(f"\nBackbone params: {total/1e6:.2f}M")
    print(f"P3={p3.shape}, P4={p4.shape}, P5={p5.shape}")

    # Test registration
    register_efficientvit_modules()
    print("\nRegistration successful!")
