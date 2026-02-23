#!/usr/bin/env python3
"""
IQ-9075 Video Inference — ONNX / QNN Context Binary.

Runs DDRNet23-Slim on Qualcomm IQ-9075 NPU via ONNX Runtime (QNN EP)
or falls back to CPU/CUDA. Reads an MP4 video and writes a segmentation
result MP4.

Supports two model formats:
  - .onnx  : ONNX model (FP32 or INT8 QDQ)
  - .bin   : QNN Context Binary (compiled for HTP/NPU)

Preprocessing (done on CPU, NOT inside the model):
  1. Resize frame to 640x544 (W x H)
  2. Convert to float [0, 1]
  3. Normalize with ImageNet mean/std

Usage on IQ-9075:
    # ONNX model (QNN EP → NPU)
    python scripts/infer_qnn_video.py \
        --model deploy/ddrnet23_slim_unified7class_544x640.onnx \
        --input video.mp4 --output result.mp4

    # QNN Context Binary (direct NPU)
    python scripts/infer_qnn_video.py \
        --model deploy/ddrnet23_slim_int8.bin \
        --input video.mp4 --output result.mp4

    # With overlay blending
    python scripts/infer_qnn_video.py \
        --model deploy/ddrnet23_slim_int8.bin \
        --input video.mp4 --output result.mp4 --overlay --alpha 0.5

Usage on host PC (for testing without IQ-9075):
    python scripts/infer_qnn_video.py \
        --model deploy/ddrnet23_slim_unified7class_544x640.onnx \
        --input video.mp4 --output result.mp4 --backend cpu
"""

import os
import sys
import argparse
import time

import numpy as np
import cv2


# ===================================================================
# 7-Class Ontology (must match training)
# ===================================================================
NUM_CLASSES = 7

CLASS_NAMES = [
    "Smooth Ground",   # 0
    "Rough Ground",    # 1
    "Vegetation",      # 2
    "Obstacle",        # 3
    "Water",           # 4
    "Sky",             # 5
    "Dynamic",         # 6
]

CLASS_COLORS = [
    (128, 64, 128),    # 0: Smooth Ground -- purple-gray
    (140, 100, 40),    # 1: Rough Ground  -- brown
    (0, 180, 0),       # 2: Vegetation    -- green
    (220, 20, 60),     # 3: Obstacle      -- crimson red
    (0, 100, 255),     # 4: Water         -- blue
    (70, 130, 180),    # 5: Sky           -- steel blue
    (255, 255, 0),     # 6: Dynamic       -- yellow
]

# Navigation costmap values (for ROS costmap_2d)
COSTMAP_VALUES = {
    0: 0,      # Smooth Ground -> FREE
    1: 80,     # Rough Ground  -> CAUTION
    2: 40,     # Vegetation    -> LOW COST
    3: 254,    # Obstacle      -> LETHAL
    4: 254,    # Water         -> LETHAL
    5: 0,      # Sky           -> FREE (ignored)
    6: 254,    # Dynamic       -> LETHAL
}

# ImageNet normalization (must match training dataset.py)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ===================================================================
# ONNX Runtime Session Builder
# ===================================================================

def create_session(model_path, backend="auto"):
    """Create ONNX Runtime inference session.

    Backend priority (auto):
      1. QNN HTP (NPU)  — IQ-9075 Hexagon NPU
      2. CUDA            — NVIDIA GPU (host PC testing)
      3. CPU             — fallback

    Args:
        model_path: path to .onnx or .bin file
        backend: "auto", "qnn", "cuda", or "cpu"

    Returns:
        ort.InferenceSession
    """
    import onnxruntime as ort

    ext = os.path.splitext(model_path)[1].lower()

    if backend == "auto":
        available = ort.get_available_providers()
        if "QNNExecutionProvider" in available:
            backend = "qnn"
        elif "CUDAExecutionProvider" in available:
            backend = "cuda"
        else:
            backend = "cpu"

    print(f"[Session] Backend: {backend}")
    print(f"[Session] Model:   {model_path}")

    if backend == "qnn":
        # QNN Execution Provider — runs on IQ-9075 HTP (NPU)
        qnn_options = {
            "backend_path": "libQnnHtp.so",
            "htp_performance_mode": "sustained_high_performance",
            "htp_graph_finalization_optimization_mode": "3",
        }

        if ext == ".bin":
            # QNN Context Binary — pre-compiled for NPU
            qnn_options["session.qnn_context_binary"] = model_path
            print(f"[Session] Loading QNN Context Binary (pre-compiled NPU)")
            session = ort.InferenceSession(
                model_path,
                providers=[("QNNExecutionProvider", qnn_options)],
            )
        else:
            # ONNX model — QNN will compile on first run
            print(f"[Session] ONNX via QNN EP (will compile on first run)")
            session = ort.InferenceSession(
                model_path,
                providers=[
                    ("QNNExecutionProvider", qnn_options),
                    "CPUExecutionProvider",
                ],
            )

    elif backend == "cuda":
        session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    else:
        session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

    # Print session info
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    print(f"[Session] Input:  {inputs[0].name} {inputs[0].shape}")
    print(f"[Session] Output: {outputs[0].name} {outputs[0].shape}")

    return session


# ===================================================================
# Preprocessing (CPU — matches training validation pipeline)
# ===================================================================

def preprocess(frame_bgr, deploy_h=544, deploy_w=640):
    """Preprocess camera frame for model input.

    This MUST match the validation pipeline in dataset.py:
      1. Resize to (deploy_w, deploy_h)
      2. Convert to float32 [0, 1]
      3. Normalize with ImageNet mean/std
      4. Transpose to NCHW

    The model does NOT include normalization internally.
    Skipping this step will produce garbage output.

    Args:
        frame_bgr: (H, W, 3) uint8 BGR from OpenCV
        deploy_h: model input height (544)
        deploy_w: model input width (640)

    Returns:
        input_tensor: (1, 3, deploy_h, deploy_w) float32 normalized
        orig_size: (orig_h, orig_w) for upsampling
    """
    orig_h, orig_w = frame_bgr.shape[:2]

    # BGR -> RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Resize to model input size
    frame_resized = cv2.resize(
        frame_rgb, (deploy_w, deploy_h),
        interpolation=cv2.INTER_LINEAR,
    )

    # uint8 [0,255] -> float32 [0,1]
    tensor = frame_resized.astype(np.float32) / 255.0

    # ImageNet normalize: (x - mean) / std
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD

    # HWC -> NCHW
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)

    return tensor, (orig_h, orig_w)


# ===================================================================
# Postprocessing
# ===================================================================

def postprocess(logits, orig_size=None):
    """Convert model output to class prediction map.

    Args:
        logits: (1, 7, H, W) float32 from model
        orig_size: if given, upsample to (orig_h, orig_w)

    Returns:
        pred: (H, W) uint8 class IDs [0..6]
    """
    # argmax over class dimension
    pred = np.argmax(logits[0], axis=0).astype(np.uint8)

    if orig_size is not None:
        orig_h, orig_w = orig_size
        pred = cv2.resize(pred, (orig_w, orig_h),
                          interpolation=cv2.INTER_NEAREST)

    return pred


def colorize(pred):
    """Convert class prediction to RGB color image."""
    h, w = pred.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, rgb in enumerate(CLASS_COLORS):
        color[pred == cid] = rgb
    return color


def to_costmap(pred):
    """Convert class prediction to navigation costmap."""
    costmap = np.full_like(pred, 255, dtype=np.uint8)
    for cid, cost in COSTMAP_VALUES.items():
        costmap[pred == cid] = cost
    return costmap


def build_legend(width, height=40):
    """Build a color legend bar for the output video."""
    n = NUM_CLASSES
    w_per = width // n
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    for i, (name, rgb) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        x0 = i * w_per
        x1 = x0 + w_per if i < n - 1 else width
        bar[:, x0:x1] = rgb
        cx = x0 + 4
        cy = height // 2 + 5
        cv2.putText(bar, name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(bar, name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return bar


# ===================================================================
# Video Processing
# ===================================================================

def process_video(session, input_path, output_path, deploy_size,
                  overlay=False, alpha=0.5, save_costmap=False):
    """Process MP4 video frame-by-frame with ONNX Runtime."""

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    deploy_h, deploy_w = deploy_size

    input_name = session.get_inputs()[0].name

    print(f"\n[Video] {input_path}")
    print(f"  Source:    {orig_w}x{orig_h} @ {fps:.1f} FPS, {total_frames} frames")
    print(f"  Model in:  {deploy_w}x{deploy_h}")
    print(f"  Overlay:   {'ON' if overlay else 'OFF'} (alpha={alpha})")
    print(f"  Output:    {output_path}")

    # Legend bar
    legend_rgb = build_legend(orig_w, height=40)
    legend_bgr = cv2.cvtColor(legend_rgb, cv2.COLOR_RGB2BGR)

    # Video writer
    out_h = orig_h + 40
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, out_h))

    costmap_writer = None
    if save_costmap:
        costmap_path = output_path.replace(".mp4", "_costmap.mp4")
        costmap_writer = cv2.VideoWriter(
            costmap_path, fourcc, fps, (orig_w, orig_h), isColor=False
        )

    latencies = []
    bar_w = 20

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()

        # Preprocess (CPU)
        input_tensor, orig_size = preprocess(frame, deploy_h, deploy_w)

        # Inference (NPU / CPU)
        logits = session.run(None, {input_name: input_tensor})[0]

        # Postprocess (CPU)
        pred = postprocess(logits, orig_size=orig_size)
        color_rgb = colorize(pred)

        ms = (time.time() - t0) * 1000
        latencies.append(ms)

        # Build output frame
        color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
        if overlay:
            vis = cv2.addWeighted(frame, 1.0 - alpha, color_bgr, alpha, 0)
        else:
            vis = color_bgr

        out_frame = np.vstack([vis, legend_bgr])
        writer.write(out_frame)

        if costmap_writer is not None:
            costmap_writer.write(to_costmap(pred))

        # Progress
        n = frame_idx + 1
        pct = n / total_frames
        filled = int(bar_w * pct)
        bar = "\u2588" * filled + " " * (bar_w - filled)
        avg_ms = np.mean(latencies[-100:])
        print(f"\r  |{bar}| {n}/{total_frames} "
              f"{avg_ms:.1f}ms ({1000/avg_ms:.0f}FPS)  ",
              end="", flush=True)

    cap.release()
    writer.release()
    if costmap_writer is not None:
        costmap_writer.release()

    # Final summary
    avg_ms = np.mean(latencies[1:]) if len(latencies) > 1 else latencies[0]
    print(f"\r  |{'█' * bar_w}| {total_frames}/{total_frames} "
          f"avg {avg_ms:.1f}ms ({1000/avg_ms:.0f}FPS)  ")

    print(f"\n[Done] {len(latencies)} frames processed")
    print(f"  Average:  {avg_ms:.1f} ms/frame ({1000/avg_ms:.1f} FPS)")
    print(f"  Output:   {output_path}")
    if save_costmap:
        print(f"  Costmap:  {costmap_path}")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="IQ-9075 Video Inference — ONNX / QNN Context Binary"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to .onnx or .bin (QNN Context Binary)"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input MP4 video path"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output MP4 path (default: <input>_seg.mp4)"
    )
    parser.add_argument(
        "--deploy_size", type=str, default="544,640",
        help="Model input size 'H,W' (default: 544,640)"
    )
    parser.add_argument(
        "--backend", type=str, default="auto",
        choices=["auto", "qnn", "cuda", "cpu"],
        help="Inference backend (default: auto-detect)"
    )
    parser.add_argument("--overlay", action="store_true",
                        help="Blend segmentation on original frame")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Overlay blend ratio (0=original, 1=seg)")
    parser.add_argument("--save_costmap", action="store_true",
                        help="Also save navigation costmap video")
    args = parser.parse_args()

    # Parse deploy size
    parts = args.deploy_size.split(",")
    deploy_size = (int(parts[0]), int(parts[1]))

    # Default output path
    if args.output is None:
        base = os.path.splitext(args.input)[0]
        args.output = f"{base}_seg.mp4"

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  IQ-9075 Video Inference")
    print(f"{'='*60}")

    # Create session
    session = create_session(args.model, backend=args.backend)

    # Process video
    process_video(
        session, args.input, args.output, deploy_size,
        overlay=args.overlay, alpha=args.alpha,
        save_costmap=args.save_costmap,
    )


if __name__ == "__main__":
    main()