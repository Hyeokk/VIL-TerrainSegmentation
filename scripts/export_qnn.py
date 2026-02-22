#!/usr/bin/env python3
"""
Export DDRNet23-Slim to Qualcomm QNN for IQ-9075 NPU deployment.

Supports two export paths:
  A. Qualcomm AI Hub (cloud) — recommended, no local SDK needed
  B. QNN SDK (local) — requires Qualcomm QNN SDK installed

Usage:
    conda activate offroad

    # Option A: Qualcomm AI Hub (cloud compile + INT8 quantize)
    python scripts/export_qnn.py \
        --checkpoint ./checkpoints/ddrnet23-slim/best_model.pth \
        --method hub \
        --device "Qualcomm QCS9075 (Proxy)"

    # Option B: ONNX export only (for local QNN SDK conversion)
    python scripts/export_qnn.py \
        --checkpoint ./checkpoints/ddrnet23-slim/best_model.pth \
        --method onnx

    # Option C: Full local QNN conversion (requires QNN SDK)
    python scripts/export_qnn.py \
        --checkpoint ./checkpoints/ddrnet23-slim/best_model.pth \
        --method local
"""

import os
import sys
import argparse

sys.path.append(".")

import torch


def export_onnx_only(checkpoint_path, output_dir, input_size):
    """Step 1: Export trained model to ONNX.

    Creates a clean ONNX file optimized for QNN conversion:
    - Fixed input shape (no dynamic axes for H/W)
    - Opset 17 (QNN recommended)
    - Constant folding enabled
    """
    from src.models_ddrnet import build_ddrnet, load_checkpoint, export_onnx
    from src.dataset import NUM_CLASSES

    # Build model architecture (no pretrained needed, loading checkpoint)
    model = build_ddrnet(num_classes=NUM_CLASSES, pretrained=False)
    model = load_checkpoint(model, checkpoint_path)
    model.eval()

    h, w = input_size
    onnx_path = os.path.join(
        output_dir,
        f"ddrnet23_slim_unified{NUM_CLASSES}class_{h}x{w}.onnx"
    )
    os.makedirs(output_dir, exist_ok=True)

    export_onnx(model, onnx_path, input_size=input_size, opset=17)

    # Simplify ONNX
    try:
        import onnxsim
        import onnx
        print(f"\n[ONNX] Simplifying with onnxsim...")
        onnx_model = onnx.load(onnx_path)
        simplified, ok = onnxsim.simplify(onnx_model)
        if ok:
            onnx.save(simplified, onnx_path)
            print(f"[ONNX] Simplified successfully")
        else:
            print(f"[ONNX] Simplification failed, using original")
    except ImportError:
        print(f"[ONNX] onnxsim not installed, skipping simplification")

    return onnx_path


def export_via_hub(checkpoint_path, output_dir, input_size, device_name):
    """Export via Qualcomm AI Hub (cloud compile + INT8 quantization).

    This is the recommended path:
    1. Export to ONNX locally
    2. Upload to AI Hub
    3. Compile for target device with INT8 quantization
    4. Download compiled QNN Context Binary
    """
    # Step 1: Export ONNX
    onnx_path = export_onnx_only(checkpoint_path, output_dir, input_size)

    # Step 2: Try AI Hub compile
    try:
        import qai_hub as hub

        print(f"\n{'='*60}")
        print(f"  Qualcomm AI Hub Compilation")
        print(f"{'='*60}")
        print(f"  Device: {device_name}")
        print(f"  Runtime: QNN Context Binary (INT8)")
        print(f"  ONNX: {onnx_path}")
        print()

        # Upload model
        print("[Hub] Uploading ONNX model...")
        model = hub.upload_model(onnx_path)

        # Compile with INT8 quantization for IQ-9075
        h, w = input_size
        print(f"[Hub] Compiling for {device_name}...")
        compile_job = hub.submit_compile_job(
            model=model,
            device=hub.Device(device_name),
            input_specs=dict(image=(1, 3, h, w)),
            options="--target_runtime qnn_context_binary --quantize_full_type int8",
        )

        # Wait and download
        print(f"[Hub] Waiting for compilation (may take several minutes)...")
        print(f"[Hub] Job URL: {compile_job.url}")

        target_model = compile_job.get_target_model()

        output_bin = os.path.join(output_dir, "ddrnet23_slim_int8.bin")
        target_model.download(output_bin)
        print(f"\n[Hub] QNN Context Binary saved: {output_bin}")
        print(f"[Hub] Ready to deploy to IQ-9075!")

        # Profile on device
        print(f"\n[Hub] Profiling on {device_name}...")
        profile_job = hub.submit_profile_job(
            model=target_model,
            device=hub.Device(device_name),
        )
        print(f"[Hub] Profile URL: {profile_job.url}")

    except ImportError:
        print(f"\n[Hub] qai_hub not installed.")
        print(f"[Hub] Install: pip install qai-hub")
        print(f"[Hub] Then configure: qai-hub configure --api_token YOUR_TOKEN")
        print(f"\n[Hub] ONNX file is ready for manual upload at: {onnx_path}")
        _print_manual_hub_instructions(onnx_path, device_name, input_size)

    except Exception as e:
        print(f"\n[Hub] Error during compilation: {e}")
        print(f"[Hub] ONNX file is ready for manual upload at: {onnx_path}")
        _print_manual_hub_instructions(onnx_path, device_name, input_size)

    return onnx_path


def export_local_qnn(checkpoint_path, output_dir, input_size):
    """Export using local QNN SDK (SNPE/QNN tools required).

    Pipeline:
    1. Export to ONNX
    2. Convert ONNX → QNN model library (.so)
    3. Quantize to INT8 using calibration data
    4. Create QNN Context Binary
    """
    onnx_path = export_onnx_only(checkpoint_path, output_dir, input_size)

    print(f"\n{'='*60}")
    print(f"  Local QNN SDK Conversion Commands")
    print(f"{'='*60}")
    print()

    model_tag = "ddrnet23_slim_7class"
    h, w = input_size

    print("  Step 1: ONNX → QNN Model Library")
    print("  ─────────────────────────────────")
    print(f"  qnn-onnx-converter \\")
    print(f"      --input_network {onnx_path} \\")
    print(f"      --output_path {output_dir}/{model_tag}.cpp")
    print()

    print("  Step 2: Compile to shared library")
    print("  ─────────────────────────────────")
    print(f"  qnn-model-lib-generator \\")
    print(f"      -c {output_dir}/{model_tag}.cpp \\")
    print(f"      -b {output_dir}/{model_tag}.bin \\")
    print(f"      -o {output_dir}/{model_tag}.so")
    print()

    print("  Step 3: Generate calibration data")
    print("  ─────────────────────────────────")
    print(f"  python scripts/generate_calibration.py \\")
    print(f"      --num_samples 200 \\")
    print(f"      --input_size {h},{w} \\")
    print(f"      --output_dir {output_dir}/calibration/")
    print()

    print("  Step 4: Quantize to INT8")
    print("  ─────────────────────────────────")
    print(f"  qnn-net-run \\")
    print(f"      --model {output_dir}/{model_tag}.so \\")
    print(f"      --input_list {output_dir}/calibration/input_list.txt \\")
    print(f"      --output_dir {output_dir}/quantized/ \\")
    print(f"      --profiling_level basic")
    print()

    print("  Step 5: Create QNN Context Binary")
    print("  ─────────────────────────────────")
    print(f"  qnn-context-binary-generator \\")
    print(f"      --model {output_dir}/{model_tag}.so \\")
    print(f"      --backend libQnnHtp.so \\")
    print(f"      --output_dir {output_dir}/context_binary/")
    print()

    return onnx_path


def _print_manual_hub_instructions(onnx_path, device_name, input_size):
    """Print instructions for manual Qualcomm AI Hub usage."""
    h, w = input_size

    print(f"\n{'='*60}")
    print(f"  Manual Qualcomm AI Hub Instructions")
    print(f"{'='*60}")
    print()
    print(f"  1. Go to https://aihub.qualcomm.com")
    print(f"  2. Upload: {onnx_path}")
    print(f"  3. Select device: {device_name}")
    print(f"  4. Set target runtime: QNN Context Binary")
    print(f"  5. Set quantization: INT8 (W8A8)")
    print(f"  6. Set input spec: image=(1, 3, {h}, {w})")
    print()
    print(f"  Or use the Python API:")
    print(f"  ─────────────────────────────────")
    print(f"  import qai_hub as hub")
    print(f"  model = hub.upload_model('{onnx_path}')")
    print(f"  job = hub.submit_compile_job(")
    print(f"      model=model,")
    print(f"      device=hub.Device('{device_name}'),")
    print(f"      input_specs=dict(image=(1, 3, {h}, {w})),")
    print(f"      options='--target_runtime qnn_context_binary "
          f"--quantize_full_type int8',")
    print(f"  )")
    print(f"  target = job.get_target_model()")
    print(f"  target.download('ddrnet23_slim_int8.bin')")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Export DDRNet23-Slim for Qualcomm IQ-9075 NPU"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="./checkpoints/ddrnet23-slim/best_model.pth",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--method", type=str, default="onnx",
        choices=["hub", "onnx", "local"],
        help="Export method: 'hub' (AI Hub cloud), 'onnx' (ONNX only), 'local' (QNN SDK)"
    )
    parser.add_argument(
        "--device", type=str,
        default="Qualcomm QCS9075 (Proxy)",
        help="Target device name for AI Hub"
    )
    parser.add_argument(
        "--deploy_size", type=str, default="544,640",
        help="Input size as 'H,W' (default: 544,640 for S10 Ultra half-res)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./deploy",
        help="Output directory for exported files"
    )
    args = parser.parse_args()

    # Parse deploy size
    parts = args.deploy_size.split(",")
    input_size = (int(parts[0]), int(parts[1]))

    print(f"\n{'='*60}")
    print(f"  DDRNet23-Slim → IQ-9075 Export")
    print(f"{'='*60}")
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  Method:      {args.method}")
    print(f"  Input size:  {input_size[0]}×{input_size[1]} (H×W)")
    print(f"  Target:      {args.device}")
    print(f"  Output dir:  {args.output_dir}")
    print()

    if args.method == "hub":
        export_via_hub(
            args.checkpoint, args.output_dir,
            input_size, args.device
        )
    elif args.method == "onnx":
        export_onnx_only(
            args.checkpoint, args.output_dir,
            input_size
        )
    elif args.method == "local":
        export_local_qnn(
            args.checkpoint, args.output_dir,
            input_size
        )

    print(f"\n{'='*60}")
    print(f"  Deployment Pipeline Summary")
    print(f"{'='*60}")
    print(f"  Trained model      -> {args.checkpoint}")
    print(f"  ONNX (FP32)        -> {args.output_dir}/")
    print(f"  QNN INT8 (IQ-9075) -> Convert via AI Hub or QNN SDK")
    print()
    print(f"  On-device inference flow:")
    print(f"  S10 Ultra (1280x1080)")
    print(f"    -> CPU Resize ({input_size[1]}x{input_size[0]})")
    print(f"    -> NPU DDRNet INT8 (~25ms)")
    print(f"    -> argmax -> 7-class costmap")
    print(f"    -> Navigation decision")
    print()


if __name__ == "__main__":
    main()