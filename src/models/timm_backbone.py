"""
Adapter that wraps a ``timm`` backbone as an Ultralytics-compatible
multi-stage module, so MobileViT-S and EfficientFormerV2-S2 can be used
as drop-in YOLOv8 backbones for the competitor-family comparison.

Each stage is exposed as a separate ``nn.Module`` so the Ultralytics
YAML parser tracks the right number of layers and the PANet neck can
take features at strides 8 / 16 / 32 (P3 / P4 / P5).

Three quirks worked around here:

1. *SDPA backward re-entry*: PyTorch's memory-efficient SDPA backend
   chokes when multiple downstream consumers backpropagate through one
   shared backbone forward (which is exactly the topology produced by
   our TimmStem + 3x TimmStage layout). We disable mem-efficient SDPA
   at import time so PyTorch falls back to the math backend.

2. *Fixed attention-token counts*: EfficientFormer / EfficientFormerV2
   build their attention biases for a specific input resolution (default
   224). At our 640 training resolution the buffer shape is wrong. We
   pass ``img_size=640`` to the timm constructor so the buffers are
   sized correctly up front.

3. *Cache cross-iteration leak*: ``_FEATURE_CACHE`` is keyed by the
   input-tensor id, which Python may reuse across iterations. We clear
   the cache before populating it inside ``TimmStem.forward`` so each
   training step sees only its own feature maps. Cached tensors are
   cloned before being returned by ``TimmStage`` so each consumer holds
   an independent reference that survives autograd's saved-tensor
   bookkeeping.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# --- Fix #1: stop PyTorch from using SDPA backends that re-enter badly.
# Math SDPA is slower but its custom autograd survives multiple downstream
# consumers, which is exactly the topology we build here.
try:
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
except AttributeError:
    pass


# Single-process cache: full backbone forward is computed once per input
# tensor identity, then each TimmStage reads its slice from this dict.
_FEATURE_CACHE: dict[int, list[torch.Tensor]] = {}


class _TimmBackboneRunner(nn.Module):
    """Holds the full timm backbone and runs it once per forward pass."""

    def __init__(self, model_name: str, pretrained: bool = True,
                 out_indices: tuple | None = None, img_size: int = 640):
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "timm is required for MobileViT/EfficientFormer baselines. "
                "Install with: pip install timm"
            ) from e

        # Discover total stage count to pick the last 4 by default.
        probe = timm.create_model(model_name, pretrained=False, features_only=True)
        n_stages = len(probe.feature_info.channels())
        del probe
        if out_indices is None:
            out_indices = tuple(range(max(0, n_stages - 4), n_stages))

        # --- Fix #2: pass img_size for backbones with fixed-resolution
        # attention biases (EfficientFormer, EfficientFormerV2). Other
        # backbones ignore the kwarg via **kwargs.
        extra_kwargs = {}
        if "efficientformer" in model_name.lower():
            extra_kwargs["img_size"] = img_size
            # When img_size differs from the pretrained value, attention
            # biases are size-mismatched. Disable pretrained for these
            # so the model trains from scratch (still fair given the
            # YOLOv8 head also trains from scratch).
            if pretrained and img_size != 224:
                print(f"[timm-backbone] {model_name}: img_size={img_size} != 224, "
                      f"falling back to random init (attention-bias size mismatch).")
                pretrained = False

        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=out_indices,
                **extra_kwargs,
            )
        except TypeError:
            # Backbone doesn't accept our extras — retry without them.
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=out_indices,
            )
        self.feature_channels = self.backbone.feature_info.channels()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # --- Fix #3: clear before populate, so a stale id-collision can
        # never return last iteration's freed tensors.
        _FEATURE_CACHE.clear()
        feats = self.backbone(x)
        _FEATURE_CACHE[id(x)] = feats
        return feats


class TimmStem(nn.Module):
    """Layer 0: instantiate the backbone runner and emit the stride-4 feature.

    Ultralytics auto-prepends ``ch[f]`` to every YAML module's args, so the
    actual call is ``TimmStem(c1, c2, model_name, pretrained)``. ``c2`` is
    the declared stride-4 output channel count (used by Ultralytics'
    channel tracker; runtime channels come from the timm model).
    """

    def __init__(self, *args, model_name: str = "mobilevit_s",
                 pretrained: bool = True):
        super().__init__()
        # Robust to whether Ultralytics prepended c1 or not: the model name is
        # the string arg, pretrained is the bool arg.
        strs = [a for a in args if isinstance(a, str)]
        bools = [a for a in args if isinstance(a, bool)]
        if strs:
            model_name = strs[0]
        if bools:
            pretrained = bools[0]
        self.runner = _TimmBackboneRunner(model_name, pretrained=pretrained)
        self._out_idx = 0  # stride-4 feature
        self.feature_channels = self.runner.feature_channels

    def forward(self, x):
        feats = self.runner(x)
        # Clone so the downstream layer holds an independent tensor
        # reference. Autograd still routes back through the shared
        # backbone forward exactly once.
        return feats[self._out_idx].clone()


class TimmStage(nn.Module):
    """Reads a precomputed feature map from the shared runner.

    Ultralytics prepends ``c1`` to YAML args, so the call is
    ``TimmStage(c1, c2, stage_idx)``. ``stage_idx`` selects which cached
    feature map to return (0=P2, 1=P3, 2=P4, 3=P5).
    """

    def __init__(self, *args, stage_idx: int = 0):
        super().__init__()
        # stage_idx is the last integer YAML arg, robust to whether Ultralytics
        # prepended c1: (c1, c2, stage_idx) or (c2, stage_idx).
        ints = [a for a in args if isinstance(a, int) and not isinstance(a, bool)]
        if ints:
            stage_idx = ints[-1]
        self.stage_idx = stage_idx
        self._noop = nn.Identity()

    def forward(self, x):
        cache = next(iter(_FEATURE_CACHE.values()), None)
        if cache is None:
            raise RuntimeError(
                "TimmStage forward called before TimmStem populated the cache. "
                "Check the YAML layer order."
            )
        # Clone for the same reason as TimmStem above.
        return cache[self.stage_idx].clone()


def register_timm_backbone_modules():
    """Register timm backbone adapter modules with Ultralytics' parse_model.

    Inserts TimmStem/TimmStage into parse_model's ``base_modules`` set so that
    Ultralytics prepends the input-channel arg and tracks output channels for
    these layers. Robust across Ultralytics versions: a generic regex inserts
    the names into the ``base_modules`` set regardless of its other members, and
    we raise loudly if patching ever fails (instead of silently continuing).
    """
    import inspect
    import re
    import ultralytics.nn.tasks as tasks

    tasks.TimmStem = TimmStem
    tasks.TimmStage = TimmStage

    if getattr(tasks, "_timm_backbone_patched", False):
        print("[timm-backbone] Modules already registered")
        return

    # Read parse_model from the source FILE (pristine), not inspect.getsource on
    # the in-memory function: a prior registration (e.g. EfficientViT) re-execs
    # tasks, which misaligns getsource line numbers and corrupts the result.
    import pathlib
    try:
        full = pathlib.Path(tasks.__file__).read_text(encoding="utf-8")
        mobj = re.search(r"\ndef parse_model\(.*?(?=\ndef |\Z)", full, re.S)
        src = mobj.group(0) if mobj else inspect.getsource(tasks.parse_model)
    except Exception:
        src = inspect.getsource(tasks.parse_model)
    patched = None

    # Insert just after the opening of parse_model's `base_modules` membership
    # set (frozenset({...}) or plain {...}). We deliberately do NOT fall back to
    # anchoring on a module name like "A2C2f," because that token also appears in
    # the top-of-file `from ultralytics.nn.modules import (...)` statement, and
    # editing that import would raise ImportError on exec.
    # Insert TimmStem/TimmStage into parse_model's base-module membership set.
    # Different Ultralytics versions express it differently:
    #   * named var:   base_modules = frozenset({...}) / ({...}) / {...}
    #   * inline set:  if m in { Conv, ... }:        (e.g. 8.3.x, line ~32)
    # We insert right after the opening of the FIRST such set (count=1), which
    # is the base set that triggers the (c1, c2=args[0]) channel handling.
    for pat in (r"(base_modules\s*=\s*frozenset\(\s*[\{\(\[])",
                r"(base_modules\s*=\s*[\{\(\[])",
                r"(if m in \{)"):
        new = re.sub(pat, r"\1TimmStem, TimmStage, ", src, count=1)
        if new != src:
            patched = new
            break

    if patched is None:
        # Fallback: insert after the A2C2f member of base_modules, skipping any
        # `from ... import` line so the import statement is never edited.
        out, done = [], False
        for ln in src.splitlines(keepends=True):
            if (not done) and ("A2C2f" in ln) and ("import" not in ln):
                ln = (ln.replace("A2C2f,", "A2C2f, TimmStem, TimmStage,", 1)
                      if "A2C2f," in ln
                      else ln.replace("A2C2f", "A2C2f, TimmStem, TimmStage", 1))
                done = True
            out.append(ln)
        if done:
            patched = "".join(out)

    if patched is None:
        raise RuntimeError(
            "Could not patch Ultralytics parse_model to register TimmStem/TimmStage. "
            "Your installed ultralytics differs from what this adapter expects. "
            "Pin a compatible version (e.g. `pip install 'ultralytics==8.3.0'`) "
            "and retry, or report the installed version."
        )

    code = compile(patched, tasks.__file__, "exec")
    exec(code, tasks.__dict__)
    tasks._timm_backbone_patched = True
    print("[timm-backbone] Registered TimmStem / TimmStage with Ultralytics "
          "(SDPA: math backend only)")
