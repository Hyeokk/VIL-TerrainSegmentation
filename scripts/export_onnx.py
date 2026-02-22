#!/usr/bin/env python3
"""
Export trained model to ONNX for Qualcomm IQ-9075 deployment.

Supports all model architectures (EfficientViT, FFNet).
Includes instructions for Qualcomm AI Hub QNN conversion.

Usage:
    conda activate offroad

    # Export EfficientViT-B1
    python scripts/export_onnx.py --model efficientvit-b1 \
        --checkpoint ./checkpoints/efficientvit-b1/best_model.pth

    # Export FFNet-78S (recommended for IQ-9075)
    python scripts/export_onnx.py --model ffnet-78s \
        --checkpoint ./checkpoints/ffnet-78s/best_model.pth
"""

import os
import sys
import argparse

sys.path.insert(0, "./efficientvit")
sys.path.append(".")

import torch
from src.dataset import NUM_CLASSES
from src.models import build_model, load_checkpoint, SUPPORTED_MODELS


def main():
    parser = argparse.ArgumentParser(
        description="Export segmentation model to ONNX for IQ-9075 deployment"
    )
    parser.add_argument("--model", type=str, default="ddrnet23-slim",
                        choices=list(SUPPORTED_MODELS.keys()))
    parser.add_argument("--checkpoint", type=str,
                        default="./checkpoints/ddrnet23-slim/best_model.pth")
    parser.add_argument("--output_dir", type=str, default="./onnx")
    parser.add_argument(
        "--deploy_size", type=str, default="544,640",
        help="Input size as 'H,W' (e.g. '544,640') or single int"
    )
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version (17 recommended for QNN)")
    args = parser.parse_args()

    # Parse deploy_size
    parts = args.deploy_size.split(",")
    if len(parts) == 2:
        input_h, input_w = int(parts[0]), int(parts[1])
    else:
        input_h = input_w = int(parts[0])

    os.makedirs(args.output_dir, exist_ok=True)

    # Build model and load weights
    model = build_model(args.model, num_classes=NUM_CLASSES, pretrained=False)
    model = load_checkpoint(model, args.checkpoint)
    model.eval()

    # Generate output filename
    model_tag = args.model.replace("-", "_")
    output_path = os.path.join(
        args.output_dir,
        f"{model_tag}_unified7class_{input_h}x{input_w}.onnx"
    )

    # Export with rectangular input matching S10 Ultra deploy resolution
    dummy = torch.randn(1, 3, input_h, input_w)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
    )

    # File size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print(f"\n{'='*60}")
    print(f"  ONNX Export Complete")
    print(f"{'='*60}")
    print(f"  Model:    {args.model}")
    print(f"  Output:   {output_path}")
    print(f"  Size:     {file_size_mb:.1f} MB")
    print(f"  Input:    (batch, 3, {input_h}, {input_w})")
    print(f"  Output:   (batch, {NUM_CLASSES}, {input_h}, {input_w})")
    print(f"  Deploy:   S10 Ultra 1280x1080 -> resize to {input_w}x{input_h}")
    print(f"  Opset:    {args.opset}")
    print()

    # Print IQ-9075 deployment instructions
    print(f"{'='*60}")
    print(f"  Qualcomm IQ-9075 Deployment Guide")
    print(f"{'='*60}")
    print()
    print("  Option A: Qualcomm AI Hub (recommended)")
    print("  ─────────────────────────────────────────")
    print(f"  pip install qai-hub qai-hub-models")
    print(f"  python -c \"")
    print(f"  import qai_hub as hub")
    print(f"  model = hub.upload_model('{output_path}')")
    print(f"  compile_job = hub.submit_compile_job(")
    print(f"      model=model,")
    print(f"      device=hub.Device('Qualcomm QCS9075 (Proxy)'),")
    print(f"      options='--target_runtime qnn_context_binary --quantize_full_type int8',")
    print(f"  )")
    print(f"  compile_job.download_target_model('{model_tag}_int8.bin')\"")
    print()
    print("  Option B: QNN SDK (local)")
    print("  ─────────────────────────────────────────")
    print(f"  # 1. Convert ONNX → QNN model")
    print(f"  qnn-onnx-converter \\")
    print(f"      --input_network {output_path} \\")
    print(f"      --output_path {args.output_dir}/{model_tag}.cpp")
    print()
    print(f"  # 2. Quantize to INT8")
    print(f"  qnn-net-run --model {args.output_dir}/{model_tag}.so \\")
    print(f"      --input_list calibration_list.txt \\")
    print(f"      --output_dir {args.output_dir}/quantized/")
    print()
    print("  Option C: SNPE (legacy)")
    print("  ─────────────────────────────────────────")
    print(f"  snpe-onnx-to-dlc \\")
    print(f"      --input_network {output_path} \\")
    print(f"      --output_path {args.output_dir}/{model_tag}.dlc")
    print()


if __name__ == "__main__":
    main()