#!/usr/bin/env python3
"""
Evaluate trained model on validation/test set with per-class IoU reporting.

Usage:
    conda activate offroad
    python scripts/evaluate.py --model efficientvit-b1 --checkpoint ./checkpoints/efficientvit-b1/best_model.pth
    python scripts/evaluate.py --model ffnet-78s --checkpoint ./checkpoints/ffnet-78s/best_model.pth
"""

import os
import sys
import argparse

sys.path.insert(0, "./efficientvit")
sys.path.append(".")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import build_dataset, NUM_CLASSES, CLASS_NAMES
from src.models import build_model, load_checkpoint, SUPPORTED_MODELS


def compute_confusion_matrix(pred, label, num_classes):
    """Compute confusion matrix for a single prediction-label pair."""
    mask = (label >= 0) & (label < num_classes)
    label = label[mask]
    pred = pred[mask]
    k = (label * num_classes + pred).astype(np.int64)
    bincount = np.bincount(k, minlength=num_classes * num_classes)
    return bincount.reshape(num_classes, num_classes)


def evaluate_miou(model, dataloader, num_classes):
    """Evaluate mean IoU on a dataset."""
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.cuda()
            labels = labels.cuda()

            outputs = model(images)
            if outputs.shape[2:] != labels.shape[1:]:
                outputs = nn.functional.interpolate(
                    outputs, size=labels.shape[1:],
                    mode="bilinear", align_corners=False,
                )

            preds = outputs.argmax(dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()

            for p, g in zip(preds, labels_np):
                confusion += compute_confusion_matrix(p, g, num_classes)

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
    return miou, ious, confusion


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation model on validation/test set"
    )
    parser.add_argument("--model", type=str, default="ddrnet23-slim",
                        choices=list(SUPPORTED_MODELS.keys()),
                        help="Model architecture")
    parser.add_argument("--checkpoint", type=str,
                        default="./checkpoints/ddrnet23-slim/best_model.pth")
    parser.add_argument("--data_root", type=str, default="./data/Rellis-3D")
    parser.add_argument("--split_file", type=str,
                        default="./data/Rellis-3D/split_custom/test_30.lst")
    parser.add_argument(
        "--crop_size", type=str, default="544,640",
        help="Crop size as 'H,W' or single int"
    )
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    # Parse crop_size
    crop_parts = args.crop_size.split(",")
    if len(crop_parts) == 2:
        crop_size = (int(crop_parts[0]), int(crop_parts[1]))
    else:
        crop_size = (int(crop_parts[0]), int(crop_parts[0]))

    # Build validation dataset
    data_config = {
        "rellis_root": args.data_root,
        "rellis_split_val": args.split_file,
        "crop_size": crop_size,
    }
    test_set = build_dataset(data_config, is_train=False)
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=4,
    )

    # Build model and load checkpoint
    model = build_model(args.model, num_classes=NUM_CLASSES, pretrained=False)
    model = load_checkpoint(model, args.checkpoint)
    model = model.cuda()

    # Evaluate
    miou, ious, confusion = evaluate_miou(model, test_loader, NUM_CLASSES)

    print(f"\n{'='*50}")
    print(f"  Model:      {args.model}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Test set:   {len(test_set)} images")
    print(f"{'='*50}")
    print(f"\n  mIoU: {miou:.2f}%\n")
    print(f"  {'Class':<20s} {'IoU':>8s}")
    print(f"  {'-'*28}")
    for c_idx, iou_val in enumerate(ious):
        if not np.isnan(iou_val):
            print(f"  {CLASS_NAMES[c_idx]:<20s} {iou_val * 100:>7.1f}%")
        else:
            print(f"  {CLASS_NAMES[c_idx]:<20s}     N/A")
    print()


if __name__ == "__main__":
    main()