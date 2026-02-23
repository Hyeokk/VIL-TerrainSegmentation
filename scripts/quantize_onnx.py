#!/usr/bin/env python3
"""
Quantize DDRNet23-Slim ONNX model to INT8 QDQ format.

Generates calibration data from the training dataset and applies
static INT8 quantization using onnxruntime.quantization.

The output QDQ ONNX model can be loaded by onnxruntime-qnn on IQ-9075,
which will compile it on-device using the local QAIRT runtime.
This avoids version mismatch issues with pre-compiled context binaries.

Usage (host PC):
    conda activate offroad

    python scripts/quantize_onnx.py \
        --onnx_model deploy/ddrnet23_slim_unified7class_544x640.onnx \
        --output deploy/ddrnet23_slim_int8_qdq.onnx \
        --num_calibration 200

    Then transfer to IQ-9075:
    scp deploy/ddrnet23_slim_int8_qdq.onnx \
        ubuntu@<IP>:~/VIL-Project-AMR/amr_segmentation/models/

Requirements (host PC):
    pip install onnxruntime onnx numpy opencv-python Pillow
"""

import os
import sys
import argparse
import glob
import random

import numpy as np
import cv2

sys.path.append(".")


# ImageNet normalization (must match training)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def collect_calibration_images(data_dir, num_samples, input_h, input_w):
    """Collect calibration images from training datasets.

    Searches for images in RELLIS-3D, RUGD, and GOOSE (or their fast versions).
    """
    image_paths = []

    # Search patterns for all datasets
    patterns = [
        os.path.join(data_dir, "Rellis-3D_fast", "**", "*.jpg"),
        os.path.join(data_dir, "RUGD_fast", "**", "*.jpg"),
        os.path.join(data_dir, "GOOSE_fast", "**", "*.jpg"),
        os.path.join(data_dir, "Rellis-3D", "**", "*.jpg"),
        os.path.join(data_dir, "Rellis-3D", "**", "*.png"),
        os.path.join(data_dir, "RUGD", "**", "*.jpg"),
        os.path.join(data_dir, "RUGD", "**", "*.png"),
        os.path.join(data_dir, "GOOSE", "**", "*.jpg"),
        os.path.join(data_dir, "GOOSE", "**", "*.png"),
    ]

    for pattern in patterns:
        found = glob.glob(pattern, recursive=True)
        # Filter out label/annotation images
        found = [
            p for p in found
            if "label" not in p.lower()
            and "annotation" not in p.lower()
            and "mask" not in p.lower()
            and "id" not in os.path.basename(os.path.dirname(p)).lower()
        ]
        image_paths.extend(found)

    if not image_paths:
        print(f"[WARN] No images found in {data_dir}")
        print(f"       Generating random calibration data instead")
        return None

    # Deduplicate and sample
    image_paths = list(set(image_paths))
    random.seed(42)
    random.shuffle(image_paths)
    selected = image_paths[:num_samples]

    print(f"[Calibration] Found {len(image_paths)} images, selected {len(selected)}")

    return selected


def preprocess_image(image_path, input_h, input_w):
    """Preprocess a single image (must match training validation pipeline)."""
    img = cv2.imread(image_path)
    if img is None:
        return None

    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize
    img = cv2.resize(img, (input_w, input_h), interpolation=cv2.INTER_LINEAR)

    # Normalize
    tensor = img.astype(np.float32) / 255.0
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD

    # HWC -> NCHW
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)

    return tensor


class DataReaderFromImages:
    """onnxruntime.quantization CalibrationDataReader implementation."""

    def __init__(self, image_paths, input_name, input_h, input_w):
        self.image_paths = image_paths
        self.input_name = input_name
        self.input_h = input_h
        self.input_w = input_w
        self.index = 0

    def get_next(self):
        if self.index >= len(self.image_paths):
            return None

        while self.index < len(self.image_paths):
            path = self.image_paths[self.index]
            self.index += 1
            tensor = preprocess_image(path, self.input_h, self.input_w)
            if tensor is not None:
                return {self.input_name: tensor}

        return None

    def rewind(self):
        self.index = 0


class DataReaderFromRandom:
    """Fallback calibration data reader using random inputs."""

    def __init__(self, input_name, input_h, input_w, num_samples):
        self.input_name = input_name
        self.input_h = input_h
        self.input_w = input_w
        self.num_samples = num_samples
        self.index = 0
        np.random.seed(42)

    def get_next(self):
        if self.index >= self.num_samples:
            return None

        self.index += 1

        # Generate random normalized input (simulating ImageNet-normalized images)
        tensor = np.random.randn(1, 3, self.input_h, self.input_w).astype(np.float32)
        # Clip to reasonable range
        tensor = np.clip(tensor, -2.5, 2.5)

        return {self.input_name: tensor}

    def rewind(self):
        self.index = 0


def quantize_model(onnx_path, output_path, calibration_reader, per_channel=True):
    """Apply static INT8 quantization with QDQ format."""
    from onnxruntime.quantization import (
        quantize_static,
        QuantFormat,
        QuantType,
        CalibrationMethod,
    )

    print(f"\n[Quantization] Starting INT8 QDQ quantization...")
    print(f"  Input:       {onnx_path}")
    print(f"  Output:      {output_path}")
    print(f"  Format:      QDQ (QuantizeLinear / DequantizeLinear)")
    print(f"  Weight type: INT8")
    print(f"  Activation:  INT8")
    print(f"  Per-channel: {per_channel}")

    quantize_static(
        model_input=onnx_path,
        model_output=output_path,
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        per_channel=per_channel,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={
            "ActivationSymmetric": True,   # HTP prefers symmetric
            "WeightSymmetric": True,
        },
    )

    # Report size
    orig_size = os.path.getsize(onnx_path) / (1024 * 1024)
    quant_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n[Quantization] Complete")
    print(f"  Original:  {orig_size:.1f} MB")
    print(f"  Quantized: {quant_size:.1f} MB")
    print(f"  Ratio:     {orig_size / quant_size:.1f}x")


def verify_quantized_model(onnx_path, qdq_path, input_h, input_w):
    """Quick verification: compare FP32 vs QDQ outputs."""
    import onnxruntime as ort

    dummy = np.random.randn(1, 3, input_h, input_w).astype(np.float32)

    # FP32
    sess_fp32 = ort.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"]
    )
    inp_name = sess_fp32.get_inputs()[0].name
    out_fp32 = sess_fp32.run(None, {inp_name: dummy})[0]

    # QDQ
    sess_qdq = ort.InferenceSession(
        qdq_path, providers=["CPUExecutionProvider"]
    )
    inp_name_q = sess_qdq.get_inputs()[0].name
    out_qdq = sess_qdq.run(None, {inp_name_q: dummy})[0]

    # Compare
    pred_fp32 = np.argmax(out_fp32[0], axis=0)
    pred_qdq = np.argmax(out_qdq[0], axis=0)
    match_pct = np.mean(pred_fp32 == pred_qdq) * 100

    max_diff = np.max(np.abs(out_fp32 - out_qdq))
    mean_diff = np.mean(np.abs(out_fp32 - out_qdq))

    print(f"\n[Verification] FP32 vs INT8 QDQ (random input)")
    print(f"  Argmax match: {match_pct:.1f}%")
    print(f"  Max logit diff:  {max_diff:.4f}")
    print(f"  Mean logit diff: {mean_diff:.4f}")

    if match_pct > 90:
        print(f"  Status: GOOD (>90% pixel agreement)")
    elif match_pct > 75:
        print(f"  Status: ACCEPTABLE (>75%)")
    else:
        print(f"  Status: WARNING (<75%, check calibration data)")


def main():
    parser = argparse.ArgumentParser(
        description="Quantize DDRNet23-Slim ONNX to INT8 QDQ"
    )
    parser.add_argument(
        "--onnx_model", type=str,
        default="deploy/ddrnet23_slim_unified7class_544x640.onnx",
        help="Path to FP32 ONNX model"
    )
    parser.add_argument(
        "--output", type=str,
        default="deploy/ddrnet23_slim_int8_qdq.onnx",
        help="Output path for INT8 QDQ ONNX"
    )
    parser.add_argument(
        "--data_dir", type=str, default="./data",
        help="Training data directory for calibration images"
    )
    parser.add_argument(
        "--num_calibration", type=int, default=200,
        help="Number of calibration samples"
    )
    parser.add_argument(
        "--deploy_size", type=str, default="544,640",
        help="Input size H,W"
    )
    parser.add_argument(
        "--skip_verify", action="store_true",
        help="Skip FP32 vs QDQ verification"
    )
    args = parser.parse_args()

    parts = args.deploy_size.split(",")
    input_h, input_w = int(parts[0]), int(parts[1])

    print(f"\n{'='*60}")
    print(f"  DDRNet23-Slim INT8 QDQ Quantization")
    print(f"{'='*60}")
    print(f"  Model:        {args.onnx_model}")
    print(f"  Output:       {args.output}")
    print(f"  Input size:   {input_h}x{input_w}")
    print(f"  Calibration:  {args.num_calibration} samples")
    print(f"  Data dir:     {args.data_dir}")

    # Determine input name from ONNX model
    import onnxruntime as ort
    sess = ort.InferenceSession(
        args.onnx_model, providers=["CPUExecutionProvider"]
    )
    input_name = sess.get_inputs()[0].name
    print(f"  Input name:   {input_name}")
    del sess

    # Collect calibration data
    image_paths = collect_calibration_images(
        args.data_dir, args.num_calibration, input_h, input_w
    )

    if image_paths:
        reader = DataReaderFromImages(
            image_paths, input_name, input_h, input_w
        )
    else:
        print("[Calibration] Using random data (less accurate, but functional)")
        reader = DataReaderFromRandom(
            input_name, input_h, input_w, args.num_calibration
        )

    # Quantize
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    quantize_model(args.onnx_model, args.output, reader)

    # Verify
    if not args.skip_verify:
        verify_quantized_model(args.onnx_model, args.output, input_h, input_w)

    print(f"\n{'='*60}")
    print(f"  Next Steps")
    print(f"{'='*60}")
    print(f"  1. Transfer to IQ-9075:")
    print(f"     scp {args.output} ubuntu@<IP>:~/VIL-Project-AMR/amr_segmentation/models/")
    print(f"")
    print(f"  2. Run on IQ-9075 (first run compiles for NPU):")
    print(f"     python3 scripts/infer_video.py \\")
    print(f"         --model models/{os.path.basename(args.output)} \\")
    print(f"         --input samples/inputs/road.mp4 \\")
    print(f"         --output samples/outputs/road_result.mp4 --overlay")
    print()


if __name__ == "__main__":
    main()