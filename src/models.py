"""
Unified model factory for off-road semantic segmentation.

Supports three model families for Qualcomm IQ-9075 deployment:
  1. DDRNet23-Slim — Qualcomm NPU optimized, INT8-safe (★ RECOMMENDED)
  2. EfficientViT-Seg (B0, B1, B2) — MIT Han Lab (NOT INT8-safe)
  3. FFNet (40S, 54S, 78S) — Qualcomm AI Research

All models are loaded with pretrained backbones and their segmentation
heads are replaced to output NUM_CLASSES (7) channels.

Usage:
    from src.models import build_model, load_checkpoint

    # DDRNet23-Slim (recommended for IQ-9075)
    model = build_model("ddrnet23-slim", num_classes=7, pretrained=True)

    # EfficientViT-B1 (legacy — high accuracy, INT8 issues)
    model = build_model("efficientvit-b1", num_classes=7, pretrained=True)

    # FFNet-78S (Qualcomm AI Research)
    model = build_model("ffnet-78s", num_classes=7, pretrained=True)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os


# ===================================================================
# Model Registry
# ===================================================================

SUPPORTED_MODELS = {
    # DDRNet family — Qualcomm NPU optimized, INT8-safe
    "ddrnet23-slim": {
        "family": "ddrnet",
        "params": "5.7M",
        "cityscapes_miou": "77.8%",
        "int8_safe": True,
        "npu_verified": True,
    },
    # EfficientViT family — high accuracy, NOT INT8-safe
    "efficientvit-b0": {
        "family": "efficientvit",
        "zoo_name": "efficientvit-seg-b0-cityscapes",
        "params": "0.7M",
        "cityscapes_miou": "75.7%",
    },
    "efficientvit-b1": {
        "family": "efficientvit",
        "zoo_name": "efficientvit-seg-b1-cityscapes",
        "params": "4.8M",
        "cityscapes_miou": "80.5%",
    },
    "efficientvit-b2": {
        "family": "efficientvit",
        "zoo_name": "efficientvit-seg-b2-cityscapes",
        "params": "15M",
        "cityscapes_miou": "82.1%",
    },
    # FFNet family (Qualcomm-optimized)
    "ffnet-40s": {
        "family": "ffnet",
        "variant": "segmentation_ffnet40S_BBB_mobile_pre_down",
        "params": "13.9M",
        "cityscapes_miou": "~74%",
    },
    "ffnet-54s": {
        "family": "ffnet",
        "variant": "segmentation_ffnet54S_BBB_mobile_pre_down",
        "params": "~18M",
        "cityscapes_miou": "~76%",
    },
    "ffnet-78s": {
        "family": "ffnet",
        "variant": "segmentation_ffnet78S_BCC_mobile_pre_down",
        "params": "26.8M",
        "cityscapes_miou": "~77%",
    },
}


def list_models():
    """Print all supported models with their specifications."""
    print(f"\n{'Model':<20s} {'Params':<10s} {'Cityscapes mIoU':<18s} {'Family':<12s} {'INT8 Safe'}")
    print("-" * 80)
    for name, info in SUPPORTED_MODELS.items():
        int8 = "✅" if info.get("int8_safe", False) else "—"
        print(f"{name:<20s} {info['params']:<10s} {info['cityscapes_miou']:<18s} {info['family']:<12s} {int8}")
    print()


# ===================================================================
# DDRNet Builder (delegates to src/models_ddrnet.py)
# ===================================================================

def _build_ddrnet(model_name, num_classes, pretrained=True):
    """Build DDRNet23-Slim from qai_hub_models with head replacement.

    DDRNet23-Slim uses Conv+BN+ReLU only, making it fully INT8-safe
    for Qualcomm Hexagon NPU deployment.

    Pretrained weights are auto-downloaded by qai_hub_models.
    No manual download needed.

    Args:
        model_name: 'ddrnet23-slim'
        num_classes: number of output classes (7 for our ontology)
        pretrained: load Cityscapes pretrained weights (default: True)

    Returns:
        nn.Module with num_classes output channels
    """
    from src.models_ddrnet import build_ddrnet
    return build_ddrnet(num_classes=num_classes, pretrained=pretrained)


# ===================================================================
# EfficientViT Builder
# ===================================================================

def _build_efficientvit(model_name, num_classes, pretrained=True):
    """Build EfficientViT-Seg model and replace segmentation head.

    Pretrained weights are loaded from the project's assets/ directory.
    EfficientViT's seg_model_zoo does NOT auto-download weights.
    You must manually download and place .pt files:

        assets/efficientvit_seg_b0_cityscapes.pt
        assets/efficientvit_seg_b1_cityscapes.pt

    Download from:
        https://huggingface.co/han-cai/efficientvit-seg/resolve/main/<filename>

    Args:
        model_name: one of 'efficientvit-b0', 'efficientvit-b1', 'efficientvit-b2'
        num_classes: number of output classes (7 for our ontology)
        pretrained: load Cityscapes pretrained weights from assets/

    Returns:
        nn.Module with num_classes output channels
    """
    info = SUPPORTED_MODELS[model_name]
    zoo_name = info["zoo_name"]

    # Import from EfficientViT repository
    try:
        from efficientvit.seg_model_zoo import create_efficientvit_seg_model
    except ImportError:
        raise ImportError(
            "EfficientViT not found. Run: git clone https://github.com/mit-han-lab/efficientvit.git\n"
            "And ensure 'efficientvit' directory is in your project root or sys.path."
        )

    # Resolve pretrained weight path from project assets/ directory
    # Expected: assets/efficientvit_seg_b1_cityscapes.pt
    weight_url = None
    if pretrained:
        # Map model name to .pt filename
        weight_filenames = {
            "efficientvit-b0": "efficientvit_seg_b0_cityscapes.pt",
            "efficientvit-b1": "efficientvit_seg_b1_cityscapes.pt",
            "efficientvit-b2": "efficientvit_seg_b2_cityscapes.pt",
        }
        pt_filename = weight_filenames.get(model_name)
        if pt_filename:
            # Search in project assets/ directory (relative to project root)
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            assets_path = os.path.join(project_root, "assets", pt_filename)

            if os.path.exists(assets_path):
                weight_url = assets_path
                print(f"[EfficientViT] Pretrained weights: {assets_path}")
            else:
                raise FileNotFoundError(
                    f"Pretrained weight not found: {assets_path}\n"
                    f"Please download from:\n"
                    f"  wget -O {assets_path} "
                    f"https://huggingface.co/han-cai/efficientvit-seg/resolve/main/{pt_filename}"
                )

    model = create_efficientvit_seg_model(
        zoo_name,
        pretrained=(weight_url is not None),
        weight_url=weight_url,
    )

    # Replace the final Conv2d head: 19 (Cityscapes) → num_classes
    replaced = False
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.out_channels == 19:
            parts = name.rsplit(".", 1)
            parent = dict(model.named_modules())[parts[0]]
            attr = parts[1]
            new_conv = nn.Conv2d(
                module.in_channels,
                num_classes,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                bias=(module.bias is not None),
            )
            nn.init.xavier_uniform_(new_conv.weight)
            if new_conv.bias is not None:
                nn.init.zeros_(new_conv.bias)
            setattr(parent, attr, new_conv)
            print(f"[EfficientViT] Replaced {name}: 19 → {num_classes} classes")
            replaced = True
            break

    if not replaced:
        raise RuntimeError(
            f"Could not find Conv2d with out_channels=19 in {zoo_name}. "
            "Model structure may have changed."
        )

    return model


# ===================================================================
# FFNet Builder
# ===================================================================

class FFNetSegWrapper(nn.Module):
    """Wrapper for FFNet segmentation models.

    FFNet outputs feature maps at 1/8 or 1/4 resolution.
    This wrapper adds bilinear upsampling to match the input resolution,
    and replaces the Cityscapes 19-class head with num_classes.

    FFNet architecture:
        - ResNet-like backbone with enlarged receptive field (no dilated conv)
        - Simple Up-head with nearest/bilinear upsampling
        - Segmentation head (1x1 conv)

    This is a pure CNN model — all operations are NPU-friendly for
    Qualcomm Hexagon NPU (INT8 quantization safe).
    """

    def __init__(self, ffnet_model, num_classes, original_num_classes=19):
        super().__init__()
        self.model = ffnet_model
        self.num_classes = num_classes

        # Find and replace the segmentation head
        self._replace_seg_head(num_classes, original_num_classes)

    def _replace_seg_head(self, num_classes, original_num_classes):
        """Replace the final classification layer in FFNet."""
        replaced = False

        # FFNet typically has the seg head as the last Conv2d with out_channels=19
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and module.out_channels == original_num_classes:
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent = dict(self.model.named_modules())[parts[0]]
                    attr = parts[1]
                else:
                    parent = self.model
                    attr = parts[0]

                new_conv = nn.Conv2d(
                    module.in_channels,
                    num_classes,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    bias=(module.bias is not None),
                )
                nn.init.xavier_uniform_(new_conv.weight)
                if new_conv.bias is not None:
                    nn.init.zeros_(new_conv.bias)
                setattr(parent, attr, new_conv)
                print(f"[FFNet] Replaced {name}: {original_num_classes} → {num_classes} classes")
                replaced = True
                break

        if not replaced:
            raise RuntimeError(
                f"Could not find Conv2d with out_channels={original_num_classes} in FFNet."
            )

    def forward(self, x):
        input_size = x.shape[2:]
        out = self.model(x)

        # FFNet may output at lower resolution — upsample to input size
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)

        return out


def _build_ffnet(model_name, num_classes, pretrained=True):
    """Build FFNet segmentation model.

    FFNet models can be loaded in two ways:
      1. From Qualcomm AI Hub (qai_hub_models) — recommended for deployment
      2. From the FFNet source repository — recommended for training

    Args:
        model_name: one of 'ffnet-40s', 'ffnet-54s', 'ffnet-78s'
        num_classes: number of output classes
        pretrained: load Cityscapes pretrained weights

    Returns:
        nn.Module (FFNetSegWrapper) with num_classes output channels
    """
    info = SUPPORTED_MODELS[model_name]
    variant = info["variant"]

    # Try loading from FFNet source repository
    ffnet_path = os.path.join(os.path.dirname(__file__), "..", "FFNet")
    if not os.path.isdir(ffnet_path):
        ffnet_path = os.path.join(os.path.dirname(__file__), "..", "ffnet")

    if os.path.isdir(ffnet_path):
        sys.path.insert(0, ffnet_path)
        try:
            from models.model_registry import model_entrypoint
            create_fn = model_entrypoint(variant)
            model = create_fn(pretrained=pretrained)
            print(f"[FFNet] Loaded {variant} from FFNet repository")
            return FFNetSegWrapper(model, num_classes)
        except Exception as e:
            print(f"[FFNet] Warning: FFNet repo load failed: {e}")
            sys.path.pop(0)

    # Try loading from qai_hub_models
    try:
        if "40s" in model_name:
            from qai_hub_models.models.ffnet_40s import Model
        elif "54s" in model_name:
            from qai_hub_models.models.ffnet_54s import Model
        elif "78s" in model_name:
            from qai_hub_models.models.ffnet_78s import Model
        else:
            raise ValueError(f"Unknown FFNet variant: {model_name}")

        model = Model.from_pretrained() if pretrained else Model.from_pretrained()
        print(f"[FFNet] Loaded {model_name} from qai_hub_models")
        return FFNetSegWrapper(model, num_classes)
    except ImportError:
        pass

    # Fallback: try loading from timm + custom definition
    try:
        return _build_ffnet_from_timm(model_name, num_classes, pretrained)
    except Exception:
        pass

    raise ImportError(
        f"Could not load FFNet model '{model_name}'.\n"
        "Please install one of:\n"
        "  1. Clone FFNet repo: git clone https://github.com/Qualcomm-AI-research/FFNet.git\n"
        "  2. Install qai_hub_models: pip install qai-hub-models\n"
        "  3. Download pretrained weights manually from Qualcomm AI Hub"
    )


def _build_ffnet_from_timm(model_name, num_classes, pretrained):
    """Build a ResNet-based FFNet-like model using timm.

    This is a fallback that creates a simple encoder-decoder architecture
    mimicking FFNet's design: ResNet backbone + lightweight upsampling head.
    Use this if the official FFNet repo or qai_hub_models are not available.
    """
    import timm

    # Map model names to timm backbone configs
    timm_configs = {
        "ffnet-40s": {"backbone": "resnet34", "head_channels": 128},
        "ffnet-54s": {"backbone": "resnet50", "head_channels": 128},
        "ffnet-78s": {"backbone": "resnet50", "head_channels": 256},
    }

    if model_name not in timm_configs:
        raise ValueError(f"No timm config for {model_name}")

    config = timm_configs[model_name]
    print(f"[FFNet-fallback] Building {model_name} from timm/{config['backbone']}")

    backbone = timm.create_model(
        config["backbone"],
        pretrained=pretrained,
        features_only=True,
        out_indices=(1, 2, 3, 4),
    )

    return SimpleSegModel(
        backbone=backbone,
        in_channels_list=backbone.feature_info.channels(),
        head_channels=config["head_channels"],
        num_classes=num_classes,
    )


class SimpleSegModel(nn.Module):
    """Simple FPN-like segmentation model for FFNet fallback.

    Pure CNN architecture — fully NPU compatible.
    No attention, no dilated convolutions.
    """

    def __init__(self, backbone, in_channels_list, head_channels, num_classes):
        super().__init__()
        self.backbone = backbone

        # Lateral connections (1x1 conv to unify channel dims)
        self.laterals = nn.ModuleList([
            nn.Conv2d(ch, head_channels, 1) for ch in in_channels_list
        ])

        # Merge + upsample
        self.merge_conv = nn.Sequential(
            nn.Conv2d(head_channels, head_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True),
        )

        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(head_channels, head_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_channels, num_classes, 1),
        )

        # Initialize new layers
        for m in [self.laterals, self.merge_conv, self.seg_head]:
            for module in m.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        input_size = x.shape[2:]
        features = self.backbone(x)

        # Process from deepest to shallowest
        laterals = [lat(f) for lat, f in zip(self.laterals, features)]

        # Top-down pathway
        out = laterals[-1]
        for i in range(len(laterals) - 2, -1, -1):
            out = F.interpolate(out, size=laterals[i].shape[2:],
                                mode="bilinear", align_corners=False)
            out = out + laterals[i]

        out = self.merge_conv(out)
        out = self.seg_head(out)

        # Upsample to input resolution
        out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)
        return out


# ===================================================================
# Unified Build Function
# ===================================================================

def build_model(model_name, num_classes=7, pretrained=True):
    """Build a segmentation model by name.

    Args:
        model_name: model identifier (see SUPPORTED_MODELS)
        num_classes: number of output classes (default: 7)
        pretrained: load pretrained weights (default: True)

    Returns:
        nn.Module

    Example:
        model = build_model("ddrnet23-slim", num_classes=7)   # recommended
        model = build_model("efficientvit-b1", num_classes=7)
        model = build_model("ffnet-78s", num_classes=7)
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model: '{model_name}'. Supported models:\n"
            + "\n".join(f"  - {k}" for k in SUPPORTED_MODELS.keys())
        )

    info = SUPPORTED_MODELS[model_name]
    family = info["family"]

    int8_tag = " [INT8-safe ✅]" if info.get("int8_safe") else ""
    print(f"\n[Model] Building {model_name} ({info['params']} params, "
          f"Cityscapes {info['cityscapes_miou']}){int8_tag}")

    if family == "ddrnet":
        return _build_ddrnet(model_name, num_classes, pretrained)
    elif family == "efficientvit":
        return _build_efficientvit(model_name, num_classes, pretrained)
    elif family == "ffnet":
        return _build_ffnet(model_name, num_classes, pretrained)
    else:
        raise ValueError(f"Unknown model family: {family}")


def load_checkpoint(model, checkpoint_path, strict=True):
    """Load a checkpoint into a model, handling device mapping.

    Args:
        model: nn.Module
        checkpoint_path: path to .pth file
        strict: strict state_dict loading (default: True)

    Returns:
        model with loaded weights
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Handle case where checkpoint is wrapped in a dict
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=strict)
    print(f"[Checkpoint] Loaded weights from {checkpoint_path}")
    return model