#!/usr/bin/env python3
"""
Inference pipeline for S10 Ultra camera deployment.

Supports both image files and mp4 video input.
Output matches input: image → image, video → video.

Pipeline:
    S10 Ultra frame (1280×1080)
    → Undistort (optional)
    → Resize to 640×544 (학습 validation과 동일한 전처리)
    → Model inference → 7-class segmentation
    → Upsample to original resolution
    → Overlay / costmap

Usage:
    # Image inference
    python scripts/infer_cam.py --checkpoint ./checkpoints/efficientvit-b1/best_model.pth \
        --input frame.jpg

    # Video inference (S10 Ultra mp4 → segmentation mp4)
    python scripts/infer_cam.py --checkpoint ./checkpoints/efficientvit-b1/best_model.pth \
        --input video.mp4 --output result.mp4

    # Video with overlay blending
    python scripts/infer_cam.py --checkpoint ./checkpoints/efficientvit-b1/best_model.pth \
        --input video.mp4 --output result.mp4 --overlay --alpha 0.5
"""

import os
import sys
import argparse
import glob
import time

sys.path.insert(0, "./efficientvit")
sys.path.append(".")

import numpy as np
import torch
import cv2
from torchvision import transforms

from src.dataset import NUM_CLASSES, CLASS_NAMES, CLASS_COLORS
from src.models import build_model, load_checkpoint, SUPPORTED_MODELS


# ===================================================================
# Costmap Mapping (for ROS navigation stack)
# ===================================================================
COSTMAP_VALUES = {
    0: 0,      # Smooth Ground → FREE
    1: 80,     # Rough Ground  → CAUTION
    2: 40,     # Vegetation    → LOW COST
    3: 254,    # Obstacle      → LETHAL
    4: 254,    # Water         → LETHAL
    5: 0,      # Sky           → FREE
    6: 254,    # Dynamic       → LETHAL
}

# Video file extensions
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class InferencePipeline:
    """End-to-end inference pipeline for S10 Ultra camera deployment.

    핵심 설계 원칙:
        학습 시 validation pipeline이 모든 이미지를 (crop_h, crop_w)로 resize하여
        모델에 입력합니다. 추론 시에도 반드시 동일한 전처리를 적용해야
        학습/추론 간 입력 분포가 일치하여 최적 성능이 나옵니다.

    왜 Resize인가? (Crop이 아닌 이유):
        - 학습 시 RandomScale + RandomCrop으로 다양한 스케일을 학습
        - 검증/추론 시에는 전체 장면을 빠짐없이 봐야 하므로 Resize
        - 1280×1080 → 640×544는 거의 정확히 0.5x 축소 (종횡비 보존)
        - Crop을 하면 장면의 일부만 보게 되어 전체 맥락 손실
    """

    def __init__(self, model, deploy_size=(544, 640), calibration_file=None):
        self.model = model
        self.deploy_h, self.deploy_w = deploy_size
        self.device = next(model.parameters()).device

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        self.K = None
        self.D = None
        if calibration_file and os.path.exists(calibration_file):
            calib = np.load(calibration_file)
            self.K = calib["camera_matrix"]
            self.D = calib["dist_coeffs"]
            print(f"[Calib] Loaded from {calibration_file}")

    def preprocess(self, frame_bgr):
        """Preprocess frame — matches validation pipeline in dataset.py.

        dataset.py validation:
            image.resize((self.crop_w, self.crop_h), Image.BILINEAR)
        이 코드:
            cv2.resize(frame, (self.deploy_w, self.deploy_h), INTER_LINEAR)
        → 동일한 결과
        """
        if self.K is not None and self.D is not None:
            frame_bgr = cv2.undistort(frame_bgr, self.K, self.D)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = frame_rgb.shape[:2]

        frame_resized = cv2.resize(
            frame_rgb, (self.deploy_w, self.deploy_h),
            interpolation=cv2.INTER_LINEAR
        )

        tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
        tensor = self.normalize(tensor).unsqueeze(0)

        return tensor, (orig_h, orig_w)

    @torch.no_grad()
    def predict(self, tensor):
        tensor = tensor.to(self.device)
        logits = self.model(tensor)
        return logits.argmax(dim=1).squeeze(0).cpu().numpy()

    def colorize(self, pred):
        h, w = pred.shape
        color = np.zeros((h, w, 3), dtype=np.uint8)
        for cid, rgb in enumerate(CLASS_COLORS):
            color[pred == cid] = rgb
        return color

    def to_costmap(self, pred):
        costmap = np.full_like(pred, 255, dtype=np.uint8)
        for cid, cost in COSTMAP_VALUES.items():
            costmap[pred == cid] = cost
        return costmap

    def run_frame(self, frame_bgr, upsample=True):
        """Process a single frame. Returns dict with pred, color, costmap, ms."""
        t0 = time.time()

        tensor, orig_size = self.preprocess(frame_bgr)
        pred = self.predict(tensor)

        if upsample:
            orig_h, orig_w = orig_size
            pred = cv2.resize(pred.astype(np.uint8), (orig_w, orig_h),
                              interpolation=cv2.INTER_NEAREST)

        color = self.colorize(pred)
        costmap = self.to_costmap(pred)
        ms = (time.time() - t0) * 1000

        return {"pred": pred, "color": color, "costmap": costmap,
                "latency_ms": ms, "original_size": orig_size}


def is_video(path):
    return os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS


def build_legend(height=40):
    """Build a color legend bar showing class names and colors."""
    n = NUM_CLASSES
    w_per_class = 160
    total_w = w_per_class * n
    bar = np.zeros((height, total_w, 3), dtype=np.uint8)
    for i, (name, rgb) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        x0 = i * w_per_class
        x1 = x0 + w_per_class
        bar[:, x0:x1] = rgb
        # Put text (white with black outline for readability)
        cx = x0 + 8
        cy = height // 2 + 5
        cv2.putText(bar, name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(bar, name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return bar


def process_video(pipeline, input_path, output_path, overlay=False,
                  alpha=0.5, save_costmap=False):
    """Process mp4 video frame-by-frame and write segmentation video."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n[Video] {input_path}")
    print(f"  Resolution: {orig_w}x{orig_h}, FPS: {fps:.1f}, Frames: {total_frames}")
    print(f"  Deploy:     {pipeline.deploy_w}x{pipeline.deploy_h}")
    print(f"  Overlay:    {'ON' if overlay else 'OFF'} (alpha={alpha})")

    # Build legend bar matching output width
    legend = build_legend(height=40)
    legend_resized = cv2.resize(legend, (orig_w, 40), interpolation=cv2.INTER_LINEAR)
    legend_bgr = cv2.cvtColor(legend_resized, cv2.COLOR_RGB2BGR)

    # Determine output frame height (original + legend bar)
    out_h = orig_h + 40
    out_w = orig_w

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    # Costmap video writer (optional)
    costmap_writer = None
    if save_costmap:
        costmap_path = output_path.replace(".mp4", "_costmap.mp4")
        costmap_writer = cv2.VideoWriter(costmap_path, fourcc, fps,
                                         (orig_w, orig_h), isColor=False)

    latencies = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = pipeline.run_frame(frame, upsample=True)
        latencies.append(result["latency_ms"])

        # Build output frame
        color_bgr = cv2.cvtColor(result["color"], cv2.COLOR_RGB2BGR)

        if overlay:
            # Blend original frame with segmentation colors
            vis = cv2.addWeighted(frame, 1.0 - alpha, color_bgr, alpha, 0)
        else:
            # Side: left=original, right=segmentation (stacked horizontally → too wide)
            # Better: just show segmentation
            vis = color_bgr

        # Add legend bar at bottom
        out_frame = np.vstack([vis, legend_bgr])
        writer.write(out_frame)

        if costmap_writer is not None:
            costmap_writer.write(result["costmap"])

        frame_idx += 1
        if frame_idx % 100 == 0 or frame_idx == total_frames:
            avg_ms = np.mean(latencies[-100:])
            print(f"  [{frame_idx}/{total_frames}] {avg_ms:.1f} ms/frame "
                  f"({1000/avg_ms:.1f} FPS)")

    cap.release()
    writer.release()
    if costmap_writer is not None:
        costmap_writer.release()

    avg_ms = np.mean(latencies[1:]) if len(latencies) > 1 else latencies[0]
    print(f"\n[Done] {frame_idx} frames processed")
    print(f"  Average: {avg_ms:.1f} ms/frame ({1000/avg_ms:.1f} FPS)")
    print(f"  Output:  {output_path}")
    if save_costmap:
        print(f"  Costmap: {costmap_path}")


def process_images(pipeline, image_paths, output_dir, overlay=False,
                   alpha=0.5, save_costmap=False):
    """Process image files and save segmentation results."""
    print(f"\n[Images] {len(image_paths)} files")
    latencies = []

    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"  [SKIP] {img_path}")
            continue

        result = pipeline.run_frame(frame, upsample=True)
        latencies.append(result["latency_ms"])
        base = os.path.splitext(os.path.basename(img_path))[0]

        color_bgr = cv2.cvtColor(result["color"], cv2.COLOR_RGB2BGR)

        if overlay:
            vis = cv2.addWeighted(frame, 1.0 - alpha, color_bgr, alpha, 0)
            cv2.imwrite(os.path.join(output_dir, f"{base}_overlay.png"), vis)
        else:
            cv2.imwrite(os.path.join(output_dir, f"{base}_seg.png"), color_bgr)

        if save_costmap:
            cv2.imwrite(os.path.join(output_dir, f"{base}_costmap.png"),
                        result["costmap"])

        print(f"  {base}: {result['latency_ms']:.1f} ms")

    if latencies:
        avg = np.mean(latencies[1:]) if len(latencies) > 1 else latencies[0]
        print(f"\n  Average: {avg:.1f} ms ({1000/avg:.1f} FPS)")


def main():
    parser = argparse.ArgumentParser(
        description="S10 Ultra camera inference — image or mp4 video"
    )
    parser.add_argument("--model", type=str, default="efficientvit-b1",
                        choices=list(SUPPORTED_MODELS.keys()))
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--input", type=str, default="./samples/input",
                        help="Input image, directory, or mp4 video "
                             "(default: ./samples/input)")
    parser.add_argument("--output", type=str, default="./samples/output",
                        help="Output path (directory for images, file path for video). "
                             "Default: ./samples/output")
    parser.add_argument(
        "--deploy_size", type=str, default="544,640",
        help="Deploy resolution 'H,W' matching training crop_size "
             "(default: '544,640' for S10 Ultra half-res)"
    )
    parser.add_argument("--calibration", type=str, default=None,
                        help="Camera calibration .npz file")
    parser.add_argument("--overlay", action="store_true",
                        help="Blend segmentation colors on top of original frame")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Overlay blend ratio (0=original, 1=seg only)")
    parser.add_argument("--save_costmap", action="store_true",
                        help="Also save navigation costmap")
    args = parser.parse_args()

    # Parse deploy_size
    parts = args.deploy_size.split(",")
    deploy_size = (int(parts[0]), int(parts[1]))

    # Build model
    print(f"\nLoading {args.model}...")
    print(f"  Checkpoint: {args.checkpoint}")
    model = build_model(args.model, num_classes=NUM_CLASSES, pretrained=False)
    model = load_checkpoint(model, args.checkpoint)
    model = model.cuda().eval()
    print(f"  Deploy size: {deploy_size[0]}x{deploy_size[1]} (HxW)")

    pipeline = InferencePipeline(
        model, deploy_size=deploy_size, calibration_file=args.calibration
    )

    # Dispatch: video or images
    if os.path.isfile(args.input) and is_video(args.input):
        # --- Video mode ---
        if os.path.isdir(args.output) or args.output == "./samples/output":
            os.makedirs(args.output, exist_ok=True)
            base = os.path.splitext(os.path.basename(args.input))[0]
            args.output = os.path.join(args.output, f"{base}_seg.mp4")

        process_video(pipeline, args.input, args.output,
                      overlay=args.overlay, alpha=args.alpha,
                      save_costmap=args.save_costmap)
    else:
        # --- Image mode ---
        os.makedirs(args.output, exist_ok=True)

        if os.path.isdir(args.input):
            image_paths = []
            for ext in ["*.png", "*.jpg", "*.jpeg"]:
                image_paths.extend(glob.glob(os.path.join(args.input, ext)))
            image_paths = sorted(image_paths)
        else:
            image_paths = [args.input]

        if not image_paths:
            print(f"No images found at {args.input}")
            return

        process_images(pipeline, image_paths, args.output,
                       overlay=args.overlay, alpha=args.alpha,
                       save_costmap=args.save_costmap)


if __name__ == "__main__":
    main()