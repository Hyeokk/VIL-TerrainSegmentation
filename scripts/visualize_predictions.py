#!/usr/bin/env python3
"""
Visualize segmentation predictions with color-coded overlays and legend.

Usage:
    conda activate offroad
    python scripts/visualize_predictions.py \
        --model efficientvit-b1 \
        --checkpoint ./checkpoints/efficientvit-b1/best_model.pth \
        --image_dir  ./data/Rellis-3D/00000/pylon_camera_node/ \
        --output_dir ./results/visualization/ \
        --num_images 20
"""

import os
import sys
import argparse
import glob

sys.path.insert(0, "./efficientvit")
sys.path.append(".")

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.dataset import NUM_CLASSES, CLASS_NAMES, CLASS_COLORS
from src.models import build_model, load_checkpoint, SUPPORTED_MODELS


def preprocess_image(image_pil, deploy_size=(544, 640)):
    """Preprocess a single PIL image for inference.

    Matches the validation pipeline: direct resize to (H, W).
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if isinstance(deploy_size, int):
        deploy_h = deploy_w = deploy_size
    else:
        deploy_h, deploy_w = deploy_size

    image_resized = image_pil.resize((deploy_w, deploy_h), Image.BILINEAR)

    tensor = transforms.ToTensor()(image_resized)
    tensor = normalize(tensor)
    return tensor.unsqueeze(0), image_resized


def predict(model, image_tensor):
    """Run segmentation inference."""
    with torch.no_grad():
        image_tensor = image_tensor.cuda()
        logits = model(image_tensor)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
    return pred


def colorize_prediction(pred):
    """Convert class-ID prediction map to RGB color image."""
    h, w = pred.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in enumerate(CLASS_COLORS):
        mask = (pred == class_id)
        color_img[mask] = color
    return color_img


def visualize_single(model, image_path, output_path, deploy_size=(544, 640), alpha=0.5):
    """Generate visualization for a single image."""
    image_pil = Image.open(image_path).convert("RGB")
    image_tensor, image_cropped = preprocess_image(image_pil, deploy_size)

    pred = predict(model, image_tensor)
    color_pred = colorize_prediction(pred)

    image_np = np.array(image_cropped)
    overlay = (image_np * (1 - alpha) + color_pred * alpha).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image_np)
    axes[0].set_title("Raw Image", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("Segmentation Overlay", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(color_pred)
    axes[2].set_title("Prediction Map", fontsize=14, fontweight="bold")
    axes[2].axis("off")

    legend_patches = []
    for class_id in range(NUM_CLASSES):
        color_norm = tuple(c / 255.0 for c in CLASS_COLORS[class_id])
        patch = mpatches.Patch(
            facecolor=color_norm, edgecolor="black",
            linewidth=0.5, label=CLASS_NAMES[class_id],
        )
        legend_patches.append(patch)

    fig.legend(
        handles=legend_patches, loc="lower center",
        fontsize=10, ncol=NUM_CLASSES, frameon=True,
        fancybox=True, bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def visualize_grid(model, image_paths, output_path, deploy_size=(544, 640), alpha=0.5,
                   max_cols=4):
    """Generate a grid visualization with multiple images."""
    n = len(image_paths)
    cols = min(n, max_cols)

    fig, axes = plt.subplots(2, cols, figsize=(5 * cols, 10))
    if cols == 1:
        axes = axes.reshape(2, 1)

    for i in range(cols):
        image_pil = Image.open(image_paths[i]).convert("RGB")
        image_tensor, image_cropped = preprocess_image(image_pil, deploy_size)
        pred = predict(model, image_tensor)
        color_pred = colorize_prediction(pred)
        image_np = np.array(image_cropped)

        axes[0, i].imshow(image_np)
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel("Raw Image", fontsize=14, fontweight="bold")

        axes[1, i].imshow(color_pred)
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel("Prediction", fontsize=14, fontweight="bold")

    legend_patches = []
    for class_id in range(NUM_CLASSES):
        color_norm = tuple(c / 255.0 for c in CLASS_COLORS[class_id])
        patch = mpatches.Patch(
            facecolor=color_norm, edgecolor="black",
            linewidth=0.5, label=CLASS_NAMES[class_id],
        )
        legend_patches.append(patch)

    fig.legend(
        handles=legend_patches, loc="lower center",
        fontsize=11, ncol=NUM_CLASSES, frameon=True,
        fancybox=True, bbox_to_anchor=(0.5, -0.01),
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  Grid visualization saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize segmentation predictions with color overlays"
    )
    parser.add_argument("--model", type=str, default="efficientvit-b1",
                        choices=list(SUPPORTED_MODELS.keys()))
    parser.add_argument("--checkpoint", type=str,
                        default="./checkpoints/efficientvit-b1/best_model.pth")
    parser.add_argument("--image_dir", type=str,
                        default="./data/Rellis-3D/00000/pylon_camera_node/")
    parser.add_argument("--output_dir", type=str,
                        default="./results/visualization/")
    parser.add_argument("--num_images", type=int, default=20)
    parser.add_argument(
        "--deploy_size", type=str, default="544,640",
        help="Deploy size as 'H,W' or single int"
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    # Parse deploy_size
    parts = args.deploy_size.split(",")
    if len(parts) == 2:
        deploy_size = (int(parts[0]), int(parts[1]))
    else:
        deploy_size = (int(parts[0]), int(parts[0]))

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"Loading {args.model} from {args.checkpoint}")
    model = build_model(args.model, num_classes=NUM_CLASSES, pretrained=False)
    model = load_checkpoint(model, args.checkpoint)
    model = model.cuda()
    model.eval()

    # Collect images
    patterns = ["*.png", "*.jpg", "*.jpeg"]
    image_paths = []
    for pat in patterns:
        image_paths.extend(glob.glob(os.path.join(args.image_dir, pat)))
    image_paths = sorted(image_paths)[:args.num_images]

    if not image_paths:
        print(f"No images found in {args.image_dir}")
        return

    print(f"Visualizing {len(image_paths)} images...")

    for i, img_path in enumerate(image_paths):
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(args.output_dir, f"{base}_vis.png")
        visualize_single(model, img_path, out_path, deploy_size, args.alpha)
        print(f"  [{i+1}/{len(image_paths)}] {out_path}")

    grid_paths = image_paths[:4]
    if grid_paths:
        grid_out = os.path.join(args.output_dir, "grid_overview.png")
        visualize_grid(model, grid_paths, grid_out, deploy_size, args.alpha)

    print(f"\nDone. Results in: {args.output_dir}")


if __name__ == "__main__":
    main()