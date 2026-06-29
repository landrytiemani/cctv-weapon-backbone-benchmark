"""
Pretrained weight initialization for EfficientViT-YOLOv8.

Two sources of pretrained weights:
1. Backbone: Official MIT EfficientViT-B1 ImageNet checkpoint
2. Neck+Head: YOLOv8n COCO checkpoint (identical channel dimensions)

Official B1 weights: https://huggingface.co/han-cai/efficientvit-cls/resolve/main/efficientvit_b1_r224.pt

Weight mapping from official → our module naming:
    Official                            →  Ours (YAML layer index)
    ─────────────────────────────────────────────────────────────────
    backbone.input_stem.op_list.0.*     →  0.layers.0.* (stem conv)
    backbone.input_stem.op_list.1.main.*→  0.layers.1.* (stem DSConv)
    backbone.stages.0.op_list.*         →  1.blocks.* (local stage 1)
    backbone.stages.1.op_list.*         →  2.blocks.* (local stage 2)
    backbone.stages.2.op_list.0.*       →  3.downsample.* (attn stage 3 downsample)
    backbone.stages.2.op_list.{1+j}.*   →  3.blocks.{j}.* (attn stage 3 blocks)
    backbone.stages.3.op_list.0.*       →  4.downsample.* (attn stage 4 downsample)
    backbone.stages.3.op_list.{1+j}.*   →  4.blocks.{j}.* (attn stage 4 blocks)
"""

import torch
import torch.nn as nn
from pathlib import Path
from ultralytics import YOLO


# Official pretrained checkpoint URL
B1_WEIGHTS_URL = "https://huggingface.co/han-cai/efficientvit-cls/resolve/main/efficientvit_b1_r224.pt"
B1_WEIGHTS_FILE = "efficientvit_b1_r224.pt"


def download_b1_weights(cache_dir=None):
    """Download official EfficientViT-B1 ImageNet pretrained weights."""
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "efficientvit"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    weights_path = cache_dir / B1_WEIGHTS_FILE
    if weights_path.exists():
        print(f"[Pretrained] Using cached weights: {weights_path}")
        return weights_path

    print(f"[Pretrained] Downloading EfficientViT-B1 weights...")
    torch.hub.download_url_to_file(B1_WEIGHTS_URL, str(weights_path))
    print(f"[Pretrained] Saved to: {weights_path}")
    return weights_path


def _build_key_mapping():
    """Build the key mapping from official checkpoint → our model.

    Returns a dict of {official_key_suffix: our_key_suffix} patterns.
    The actual mapping is done by prefix replacement since the internal
    structure (conv.weight, norm.weight, etc.) is the same.
    """
    # Official uses: ConvLayer → .conv.weight, .norm.weight, .norm.bias
    # We use:        ConvLayer → .conv.weight, .norm.weight, .norm.bias (same!)
    #
    # Official uses: DSConv → .depth_conv.*, .point_conv.*
    # We use:        DSConv → .depth_conv.*, .point_conv.* (same!)
    #
    # Official uses: MBConv → .inverted_conv.*, .depth_conv.*, .point_conv.*
    # We use:        MBConv → .inverted_conv.*, .depth_conv.*, .point_conv.* (same!)
    #
    # The key difference is the PREFIX path to each module.

    prefix_map = {}

    # ── Stem (layer 0) ──
    # Official: backbone.input_stem.op_list.0.{conv,norm}.* → Our: model.0.layers.0.{conv,norm}.*
    prefix_map["backbone.input_stem.op_list.0."] = "model.0.layers.0."

    # Official: backbone.input_stem.op_list.1.main.{depth_conv,point_conv}.* → Our: model.0.layers.1.{depth_conv,point_conv}.*
    prefix_map["backbone.input_stem.op_list.1.main."] = "model.0.layers.1."

    # ── Stage 1 (layer 1): 2 MBConv blocks ──
    # Official: backbone.stages.0.op_list.{i}.main.{inverted_conv,depth_conv,point_conv}.*
    # Our:      model.1.blocks.{i}.{inverted_conv,depth_conv,point_conv}.*
    # Note: official wraps in ResidualBlock(main=MBConv, shortcut=Identity)
    # For i=0 (downsample, no residual): official has .main. prefix
    # For i>0 (residual): official also has .main. prefix
    for i in range(2):
        prefix_map[f"backbone.stages.0.op_list.{i}.main."] = f"model.1.blocks.{i}."
    # First block (downsample) may not have .main. if not wrapped in ResidualBlock
    prefix_map[f"backbone.stages.0.op_list.0."] = f"model.1.blocks.0."

    # ── Stage 2 (layer 2): 3 MBConv blocks ──
    for i in range(3):
        prefix_map[f"backbone.stages.1.op_list.{i}.main."] = f"model.2.blocks.{i}."
        prefix_map[f"backbone.stages.1.op_list.{i}."] = f"model.2.blocks.{i}."

    # ── Stage 3 (layer 3): 1 downsample MBConv + 3 EfficientViTBlocks ──
    # Downsample: backbone.stages.2.op_list.0.* → model.3.downsample.*
    prefix_map["backbone.stages.2.op_list.0.main."] = "model.3.downsample."
    prefix_map["backbone.stages.2.op_list.0."] = "model.3.downsample."

    # Attention blocks:
    # backbone.stages.2.op_list.{1+j}.context_module.main.* → model.3.blocks.{j}.context_module.*
    # backbone.stages.2.op_list.{1+j}.local_module.main.* → model.3.blocks.{j}.local_module.*
    for j in range(3):
        src_idx = j + 1
        prefix_map[f"backbone.stages.2.op_list.{src_idx}.context_module.main."] = f"model.3.blocks.{j}.context_module."
        prefix_map[f"backbone.stages.2.op_list.{src_idx}.context_module."] = f"model.3.blocks.{j}.context_module."
        prefix_map[f"backbone.stages.2.op_list.{src_idx}.local_module.main."] = f"model.3.blocks.{j}.local_module."
        prefix_map[f"backbone.stages.2.op_list.{src_idx}.local_module."] = f"model.3.blocks.{j}.local_module."

    # ── Stage 4 (layer 4): 1 downsample MBConv + 4 EfficientViTBlocks ──
    prefix_map["backbone.stages.3.op_list.0.main."] = "model.4.downsample."
    prefix_map["backbone.stages.3.op_list.0."] = "model.4.downsample."

    for j in range(4):
        src_idx = j + 1
        prefix_map[f"backbone.stages.3.op_list.{src_idx}.context_module.main."] = f"model.4.blocks.{j}.context_module."
        prefix_map[f"backbone.stages.3.op_list.{src_idx}.context_module."] = f"model.4.blocks.{j}.context_module."
        prefix_map[f"backbone.stages.3.op_list.{src_idx}.local_module.main."] = f"model.4.blocks.{j}.local_module."
        prefix_map[f"backbone.stages.3.op_list.{src_idx}.local_module."] = f"model.4.blocks.{j}.local_module."

    # Sort by prefix length (longest first) so more specific prefixes match first
    prefix_map = dict(sorted(prefix_map.items(), key=lambda x: -len(x[0])))

    return prefix_map


def _remap_key(key, prefix_map, our_state=None):
    """Remap an official checkpoint key to our model's key using prefix replacement.

    Handles the nn.Sequential vs named-submodule differences:
    - point_conv is always nn.Sequential: .conv.→.0., .norm.→.1.
    - inverted_conv/depth_conv vary:
      - Normal MBConv (ConvLayer): keeps .conv./.norm. names
      - fewer_norm MBConv (nn.Sequential): .conv.→.0., no norm
    """
    for src_prefix, dst_prefix in prefix_map.items():
        if key.startswith(src_prefix):
            remapped = dst_prefix + key[len(src_prefix):]

            # point_conv is always nn.Sequential([Conv2d, BatchNorm2d])
            remapped = remapped.replace(".point_conv.conv.", ".point_conv.0.")
            remapped = remapped.replace(".point_conv.norm.", ".point_conv.1.")

            # For fewer_norm MBConv, inverted_conv and depth_conv are
            # nn.Sequential([Conv2d, Hardswish]): .conv. → .0.
            # Check if the direct key exists; if not, try Sequential index
            if our_state is not None and remapped not in our_state:
                alt = remapped
                # Try converting ConvLayer naming to Sequential naming
                for mod in (".inverted_conv.", ".depth_conv."):
                    if mod + "conv." in alt:
                        alt = alt.replace(mod + "conv.", mod + "0.")
                    elif mod + "norm." in alt:
                        # fewer_norm blocks have no norm in inverted/depth conv
                        # skip this key
                        return None
                if alt in our_state:
                    return alt

            return remapped
    return None


def load_pretrained_backbone(yolo_model, weights_path=None):
    """Load official EfficientViT-B1 ImageNet weights into our backbone.

    Args:
        yolo_model: The nn.Sequential model from YOLO().model
        weights_path: Path to efficientvit_b1_r224.pt (auto-downloads if None)

    Returns:
        tuple: (transferred_count, total_backbone_params)
    """
    # Download weights if needed
    if weights_path is None:
        weights_path = download_b1_weights()

    print(f"\n[Pretrained] Loading EfficientViT-B1 ImageNet backbone weights...")
    checkpoint = torch.load(str(weights_path), map_location="cpu", weights_only=True)

    # The checkpoint may be a raw state_dict or wrapped in {'state_dict': ...}
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        src_state = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        src_state = checkpoint["model"]
    else:
        src_state = checkpoint  # raw state_dict

    # Build key mapping
    prefix_map = _build_key_mapping()

    # Get our model's state dict
    our_state = yolo_model.state_dict()

    transferred = 0
    matched_keys = 0
    skipped_keys = []
    shape_mismatches = []

    for src_key, src_tensor in src_state.items():
        # Skip classification head
        if src_key.startswith("head.") or src_key.startswith("classifier."):
            continue

        # Remap key
        dst_key = _remap_key(src_key, prefix_map, our_state)
        if dst_key is None:
            skipped_keys.append(src_key)
            continue

        if dst_key not in our_state:
            skipped_keys.append(f"{src_key} → {dst_key} (not in model)")
            continue

        if src_tensor.shape != our_state[dst_key].shape:
            shape_mismatches.append(
                f"{src_key} {list(src_tensor.shape)} → {dst_key} {list(our_state[dst_key].shape)}"
            )
            continue

        our_state[dst_key] = src_tensor.clone()
        transferred += src_tensor.numel()
        matched_keys += 1

    # Load updated state dict
    yolo_model.load_state_dict(our_state, strict=True)

    # Stats
    backbone_params = sum(
        p.numel() for i, layer in enumerate(yolo_model.model)
        if i <= 4 for p in layer.parameters()
    )

    print(f"[Pretrained] Backbone: transferred {matched_keys} tensors, "
          f"{transferred:,}/{backbone_params:,} params ({100*transferred/max(backbone_params,1):.1f}%)")

    if shape_mismatches:
        print(f"[Pretrained] Shape mismatches ({len(shape_mismatches)}):")
        for m in shape_mismatches[:10]:
            print(f"  {m}")

    if skipped_keys:
        print(f"[Pretrained] Skipped {len(skipped_keys)} keys (head/unmapped)")

    return transferred, backbone_params


def load_pretrained_neck_head(yolo_model):
    """Load COCO-pretrained neck + head weights from YOLOv8n.

    Our neck structure and channel dims are identical to YOLOv8n's.
    """
    print(f"\n[Pretrained] Loading YOLOv8n COCO-pretrained neck+head weights...")

    yolov8n = YOLO("yolov8n.pt")
    src_model = yolov8n.model

    # Map our layer indices to YOLOv8n layer indices
    # Only layers with learnable parameters
    layer_map = {
        7:  12,   # C2f td1: 384→128
        10: 15,   # C2f td2: 192→64
        11: 16,   # Conv down1: 64→64
        13: 18,   # C2f bu1: 192→128
        14: 19,   # Conv down2: 128→128
        16: 21,   # C2f bu2: 384→256
        17: 22,   # Detect head
    }

    our_state = yolo_model.state_dict()
    transferred = 0
    skipped = 0

    for our_idx, v8n_idx in layer_map.items():
        src_layer = src_model.model[v8n_idx]
        dst_layer = yolo_model.model[our_idx]

        src_sd = src_layer.state_dict()
        dst_sd = dst_layer.state_dict()

        for key in dst_sd:
            full_key = f"model.{our_idx}.{key}"
            if key not in src_sd:
                skipped += dst_sd[key].numel()
                continue

            if src_sd[key].shape == dst_sd[key].shape:
                our_state[full_key] = src_sd[key].clone()
                transferred += src_sd[key].numel()
            else:
                skipped += dst_sd[key].numel()

    yolo_model.load_state_dict(our_state, strict=True)

    neck_head_params = sum(
        p.numel() for i, layer in enumerate(yolo_model.model)
        if i >= 5 for p in layer.parameters()
    )

    print(f"[Pretrained] Neck+Head: {transferred:,}/{neck_head_params:,} params "
          f"({100*transferred/max(neck_head_params,1):.1f}%) from COCO")

    del yolov8n
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return transferred


def load_all_pretrained(yolo_model, backbone_weights=None, load_neck=False):
    """Load pretrained weights.

    Args:
        yolo_model: The nn.Sequential model from YOLO().model
        backbone_weights: Path to EfficientViT-B1 checkpoint (auto-downloads if None)
        load_neck: If True, also load YOLOv8n COCO neck weights. Default False.
            COCO neck weights are tuned for CSPDarknet feature distributions,
            which differ substantially from EfficientViT features — loading them
            traps the optimizer in a basin that's hard to escape with low LR.
            Random-init neck trains much better end-to-end.
    """
    bb_transferred, bb_total = load_pretrained_backbone(yolo_model, backbone_weights)

    nh_transferred = 0
    if load_neck:
        nh_transferred = load_pretrained_neck_head(yolo_model)
    else:
        print(f"\n[Pretrained] Skipping COCO neck transfer — neck+head train from scratch.")
        print(f"[Pretrained] (CSPDarknet-tuned COCO neck weights are incompatible with")
        print(f"[Pretrained]  EfficientViT feature distributions; random init trains better.)")

    total_params = sum(p.numel() for p in yolo_model.parameters())
    total_transferred = bb_transferred + nh_transferred
    print(f"\n[Pretrained] TOTAL: {total_transferred:,}/{total_params:,} params "
          f"({100*total_transferred/total_params:.1f}%) pretrained")

    return total_transferred


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from models.efficientvit_modules import register_efficientvit_modules
    register_efficientvit_modules()

    yaml_path = str(Path(__file__).parent.parent.parent / "configs" / "efficientvit_yolov8.yaml")
    model = YOLO(yaml_path, task="detect")

    print(f"\nModel layers: {len(model.model.model)}")
    for i, layer in enumerate(model.model.model):
        params = sum(p.numel() for p in layer.parameters())
        print(f"  Layer {i}: {layer.__class__.__name__:30s} {params:>10,} params")

    load_all_pretrained(model.model)
