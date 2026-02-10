#!/usr/bin/env python3
"""
Evaluate EfficientViT on RELLIS-3D (mIoU on split_custom/test_30.lst).
"""

import sys
import os
sys.path.insert(0, './efficientvit'); sys.path.append('.')  # EfficientViT source path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import Rellis3DDataset, NUM_CLASSES
from efficientvit.seg_model_zoo import create_efficientvit_seg_model


def compute_confusion_matrix(pred, label, num_classes):
    """
    pred, label: numpy arrays (H, W), values in [0..num_classes-1] or 255 (ignore).
    """
    mask = (label >= 0) & (label < num_classes)
    label = label[mask]
    pred = pred[mask]
    n = num_classes
    k = (label * n + pred).astype(np.int64)
    bincount = np.bincount(k, minlength=n * n)
    conf = bincount.reshape(n, n)
    return conf


def evaluate_miou(model, dataloader, num_classes):
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for images, labels in dataloader:
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

            preds = outputs.argmax(dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()

            for p, g in zip(preds, labels_np):
                conf = compute_confusion_matrix(p, g, num_classes)
                confusion += conf

    ious = []
    for c in range(num_classes):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        denom = tp + fp + fn
        if denom > 0:
            ious.append(tp / denom)
        else:
            ious.append(float("nan"))

    miou = np.nanmean(ious) * 100.0
    return miou, ious


def main():
    # Dataset (test split)
    test_set = Rellis3DDataset(
        data_root="./data/Rellis-3D",
        split_file="./data/Rellis-3D/split_custom/test_30.lst",
        is_train=False,
        crop_size=512,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=4,
        shuffle=False,
        num_workers=4,
    )

    # Model (same as train.py)
    model = create_efficientvit_seg_model(
        "efficientvit-seg-b0-cityscapes",
        pretrained=False,  # will load our trained weights instead
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
    model = model.cuda()

    miou, ious = evaluate_miou(model, test_loader, NUM_CLASSES)
    print(f"mIoU on test_30: {miou:.2f}%")
    for idx, iou in enumerate(ious):
        print(f"  Class {idx:2d}: IoU = {iou * 100:.2f}%")

if __name__ == "__main__":
    main()
