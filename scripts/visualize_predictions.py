#!/usr/bin/env python3
"""
Visualize EfficientViT predictions on RELLIS-3D (colorized masks).
- Loads final_model.pth
- Runs on split_custom/test_30.lst
- Saves input / GT / prediction images for manual inspection
"""

import os
import sys
sys.path.insert(0, './efficientvit'); sys.path.append('.')

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import Rellis3DDataset, NUM_CLASSES
from efficientvit.seg_model_zoo import create_efficientvit_seg_model


# Simple color map for 18 classes (BGR-ish random but distinguishable)
# You can adjust colors as you like.
COLORS = np.array([
    [  0, 128,   0],  # 0
    [128,  64, 128],  # 1
    [128,   0,   0],  # 2
    [  0,   0, 128],  # 3
    [ 70, 130, 180],  # 4
    [  0,  60, 100],  # 5
    [  0,  80, 100],  # 6
    [  0,   0,  70],  # 7
    [  0,  80,   0],  # 8
    [128,  64,   0],  # 9
    [192, 192, 192],  # 10
    [ 64,  64,  64],  # 11
    [ 64,   0,  64],  # 12
    [192,   0, 192],  # 13
    [  0, 128, 128],  # 14
    [128, 128,   0],  # 15
    [255, 255,   0],  # 16
    [255,   0, 255],  # 17
], dtype=np.uint8)


def colorize_label(label_np):
    """
    label_np: (H, W), values in [0..NUM_CLASSES-1] or 255(ignore)
    Returns: RGB image (H, W, 3)
    """
    h, w = label_np.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)

    for cls_id in range(NUM_CLASSES):
        mask = label_np == cls_id
        color[mask] = COLORS[cls_id]

    # ignore(255) remains black
    return color


def main():
    os.makedirs("./vis", exist_ok=True)

    # Dataset (test split)
    test_set = Rellis3DDataset(
        data_root="./data/Rellis-3D",
        split_file="./data/Rellis-3D/split_custom/test_30.lst",
        is_train=False,
        crop_size=512,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=2,
    )

    # Model (same as train/evaluation)
    model = create_efficientvit_seg_model(
        "efficientvit-seg-b0-cityscapes",
        pretrained=False,
    )

    # Replace head: 19 → 18
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

    # Load trained weights
    ckpt_path = "./checkpoints/final_model.pth"
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model = model.cuda().eval()

    # Note: Rellis3DDataset returns normalized tensor + label.
    # For visualization of input, we will de-normalize.
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    max_vis = 20  # number of samples to visualize
    count = 0

    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images = images.cuda()
            labels = labels.cuda()

            outputs = model(images)
            if outputs.shape[2:] != labels.shape[1:]:
                outputs = nn.functional.interpolate(
                    outputs,
                    size=labels.shape[1:],
                    mode="bilinear",
                    align_corners=False,
                )

            preds = outputs.argmax(dim=1).cpu().numpy()[0]
            gt = labels.cpu().numpy()[0]

            # De-normalize input image
            img_np = images.cpu().numpy()[0]  # (3, H, W)
            img_np = img_np * std + mean
            img_np = np.clip(img_np, 0.0, 1.0)
            img_np = (img_np * 255).astype(np.uint8)  # (3, H, W)
            img_np = np.transpose(img_np, (1, 2, 0))  # (H, W, 3)

            # Colorize GT and prediction
            gt_color = colorize_label(gt)
            pred_color = colorize_label(preds)

            # Save images
            Image.fromarray(img_np).save(f"./vis/sample_{idx:04d}_image.png")
            Image.fromarray(gt_color).save(f"./vis/sample_{idx:04d}_gt.png")
            Image.fromarray(pred_color).save(f"./vis/sample_{idx:04d}_pred.png")

            print(f"Saved ./vis/sample_{idx:04d}_*.png")

            count += 1
            if count >= max_vis:
                break


if __name__ == "__main__":
    main()
