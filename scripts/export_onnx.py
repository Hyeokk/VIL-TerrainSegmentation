#!/usr/bin/env python3
"""
Export trained EfficientViT-Seg-B0 (RELLIS-3D) to ONNX.
"""

import os
import sys
sys.path.insert(0, './efficientvit'); sys.path.append('.')

import torch
import torch.nn as nn

from src.dataset import NUM_CLASSES
from efficientvit.seg_model_zoo import create_efficientvit_seg_model


def build_model():
    # 1) Base EfficientViT-Seg-B0 (Cityscapes pretrained structure)
    model = create_efficientvit_seg_model(
        "efficientvit-seg-b0-cityscapes",
        pretrained=False,  # structure only, we will load our own weights
    )

    # 2) Replace head: 19 → 18
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.out_channels == 19:
            parent = dict(model.named_modules())[name.rsplit(".", 1)[0]]
            attr = name.rsplit(".", 1)[1]
            new_conv = nn.Conv2d(
                module.in_channels,
                NUM_CLASSES,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                bias=(module.bias is not None),
            )
            setattr(parent, attr, new_conv)
            print(f"Replaced {name}: out_channels 19 → {NUM_CLASSES}")
            break

    return model


def main():
    os.makedirs("./onnx", exist_ok=True)

    # Build model and load trained weights
    model = build_model()
    ckpt_path = "./checkpoints/final_model.pth"
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # Dummy input: (1, 3, 512, 512) – same as training crop size
    dummy = torch.randn(1, 3, 512, 512)

    onnx_path = "./onnx/efficientvit_rellis3d_512.onnx"
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=13,
        do_constant_folding=True,
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
    )

    print(f"Exported ONNX model to {onnx_path}")


if __name__ == "__main__":
    main()
