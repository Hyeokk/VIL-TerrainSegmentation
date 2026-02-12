#!/usr/bin/env python3
"""
Pre-resize all dataset images for faster training.

Problem:
    RELLIS-3D images are 1200×1920 PNG files. Every epoch, DataLoader opens
    each PNG, decodes it, then RandomScale resizes it anyway. This wastes
    CPU time on I/O and decoding of unnecessarily large images.

Solution:
    Pre-resize images to max 1024px on the long side. Since our training
    crop is 544×640, and RandomScale goes down to 0.5x, we only need
    images as small as ~1100px to cover scale=2.0 for the crop.

    Original:  1200×1920 PNG (~5MB each) → takes ~50ms to decode
    Resized:   640×1024  JPG (~100KB)    → takes ~5ms to decode
    Speedup:   ~10x faster data loading

This creates a parallel directory structure with '_fast' suffix:
    data/Rellis-3D/         → data/Rellis-3D_fast/
    data/RUGD/              → data/RUGD_fast/
    data/GOOSE/             → data/GOOSE_fast/

Labels are resized with NEAREST interpolation (preserving class IDs).
Images are saved as JPEG for maximum read speed.

Usage:
    python scripts/preprocess_datasets.py

    # Then train with pre-resized data:
    python scripts/train.py --fast
"""

import os
import sys
import glob
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image

# Maximum size on the long side
# With crop 544×640 and scale 2.0: need at least 544*2=1088 on short side
# 1024 is a good balance: covers scale ~1.8x, massive I/O speedup
MAX_LONG_SIDE = 1024


def resize_and_save(args):
    """Resize a single image-label pair and save."""
    img_src, img_dst, lbl_src, lbl_dst = args

    try:
        # --- Image: resize + save as JPEG ---
        img = Image.open(img_src).convert("RGB")
        w, h = img.size

        # Only resize if larger than MAX_LONG_SIDE
        if max(w, h) > MAX_LONG_SIDE:
            scale = MAX_LONG_SIDE / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = img.resize((new_w, new_h), Image.BILINEAR)

        os.makedirs(os.path.dirname(img_dst), exist_ok=True)
        img.save(img_dst, "JPEG", quality=95)

        # --- Label: resize with NEAREST + save as PNG ---
        if lbl_src and os.path.exists(lbl_src):
            lbl = Image.open(lbl_src)
            lw, lh = lbl.size

            if max(lw, lh) > MAX_LONG_SIDE:
                scale = MAX_LONG_SIDE / max(lw, lh)
                new_lw = int(lw * scale)
                new_lh = int(lh * scale)
                lbl = lbl.resize((new_lw, new_lh), Image.NEAREST)

            os.makedirs(os.path.dirname(lbl_dst), exist_ok=True)
            lbl.save(lbl_dst)

        return True, img_src
    except Exception as e:
        return False, f"{img_src}: {e}"


def process_rellis(data_root, output_root):
    """Pre-resize RELLIS-3D dataset."""
    tasks = []
    sequences = sorted([d for d in os.listdir(data_root)
                        if os.path.isdir(os.path.join(data_root, d)) and d.isdigit()])

    for seq in sequences:
        img_dir = os.path.join(data_root, seq, "pylon_camera_node")
        lbl_dir = os.path.join(data_root, seq, "pylon_camera_node_label_id")

        if not os.path.isdir(img_dir):
            continue

        for fname in sorted(os.listdir(img_dir)):
            if not fname.endswith(".jpg") and not fname.endswith(".png"):
                continue

            img_src = os.path.join(img_dir, fname)
            img_dst = os.path.join(output_root, seq, "pylon_camera_node",
                                   os.path.splitext(fname)[0] + ".jpg")

            lbl_name = os.path.splitext(fname)[0] + ".png"
            lbl_src = os.path.join(lbl_dir, lbl_name)
            lbl_dst = os.path.join(output_root, seq, "pylon_camera_node_label_id", lbl_name)

            if not os.path.exists(lbl_src):
                lbl_src = None
                lbl_dst = None

            tasks.append((img_src, img_dst, lbl_src, lbl_dst))

    return tasks


def process_rugd(data_root, output_root):
    """Pre-resize RUGD dataset.

    RUGD structure:
        data/RUGD/
            RUGD_frames-with-annotations/<scene>/*.png (images)
            RUGD_annotations/<scene>/*.png (color-coded labels)
    """
    tasks = []

    frames_dir = os.path.join(data_root, "RUGD_frames-with-annotations")
    annot_dir = os.path.join(data_root, "RUGD_annotations")

    if not os.path.isdir(frames_dir) or not os.path.isdir(annot_dir):
        print(f"[RUGD] Warning: RUGD_frames-with-annotations or RUGD_annotations not found in {data_root}")
        return tasks

    for scene in sorted(os.listdir(frames_dir)):
        scene_frames = os.path.join(frames_dir, scene)
        scene_annots = os.path.join(annot_dir, scene)
        if not os.path.isdir(scene_frames) or not os.path.isdir(scene_annots):
            continue

        for fname in sorted(os.listdir(scene_frames)):
            if not fname.endswith(".png"):
                continue

            img_src = os.path.join(scene_frames, fname)
            img_dst = os.path.join(output_root, "RUGD_frames-with-annotations",
                                   scene, os.path.splitext(fname)[0] + ".jpg")

            lbl_src = os.path.join(scene_annots, fname)
            lbl_dst = os.path.join(output_root, "RUGD_annotations", scene, fname)

            if not os.path.exists(lbl_src):
                lbl_src = None
                lbl_dst = None

            tasks.append((img_src, img_dst, lbl_src, lbl_dst))

    return tasks


def process_goose(data_root, output_root):
    """Pre-resize GOOSE dataset."""
    tasks = []

    # Find image/label dirs (same logic as dataset.py)
    for split in ["train", "val"]:
        candidates = [
            (os.path.join(data_root, split, "images", split),
             os.path.join(data_root, split, "labels", split)),
            (os.path.join(data_root, "images", split),
             os.path.join(data_root, "labels", split)),
        ]

        img_base = lbl_base = None
        for img_dir, lbl_dir in candidates:
            if os.path.isdir(img_dir) and os.path.isdir(lbl_dir):
                img_base = img_dir
                lbl_base = lbl_dir
                break

        if img_base is None:
            continue

        # Iterate scenes
        for scene in sorted(os.listdir(img_base)):
            scene_img = os.path.join(img_base, scene)
            scene_lbl = os.path.join(lbl_base, scene)
            if not os.path.isdir(scene_img):
                continue

            for fname in sorted(os.listdir(scene_img)):
                if not fname.endswith(".png"):
                    continue
                # Only visible images
                if "_nir.png" in fname or "_instanceids.png" in fname:
                    continue

                img_src = os.path.join(scene_img, fname)

                # Relative path preserving structure
                rel = os.path.relpath(img_src, data_root)
                img_dst = os.path.join(output_root,
                                       os.path.splitext(rel)[0] + ".jpg")

                # Find label
                if "_windshield_vis.png" in fname:
                    lbl_name = fname.replace("_windshield_vis.png", "_labelids.png")
                elif "_color.png" in fname:
                    lbl_name = fname.replace("_color.png", "_labelids.png")
                else:
                    continue

                lbl_src = os.path.join(scene_lbl, lbl_name)
                rel_lbl = os.path.relpath(lbl_src, data_root)
                lbl_dst = os.path.join(output_root, rel_lbl)

                if not os.path.exists(lbl_src):
                    lbl_src = None
                    lbl_dst = None

                tasks.append((img_src, img_dst, lbl_src, lbl_dst))

    return tasks


def main():
    global MAX_LONG_SIDE

    parser = argparse.ArgumentParser(
        description="Pre-resize dataset images for faster training"
    )
    parser.add_argument("--data_dir", default="./data",
                        help="Root data directory")
    parser.add_argument("--max_size", type=int, default=MAX_LONG_SIDE,
                        help=f"Max long side (default: {MAX_LONG_SIDE})")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel workers")
    args = parser.parse_args()

    MAX_LONG_SIDE = args.max_size

    all_tasks = []

    # RELLIS-3D
    rellis_root = os.path.join(args.data_dir, "Rellis-3D")
    if os.path.isdir(rellis_root):
        rellis_out = os.path.join(args.data_dir, "Rellis-3D_fast")
        tasks = process_rellis(rellis_root, rellis_out)
        print(f"[RELLIS-3D] {len(tasks)} files to process → {rellis_out}")
        all_tasks.extend(tasks)

        # Copy and fix split files (image paths: .png → .jpg)
        for split_dir in ["split", "split_custom"]:
            src = os.path.join(rellis_root, split_dir)
            dst = os.path.join(rellis_out, split_dir)
            if os.path.isdir(src):
                os.makedirs(dst, exist_ok=True)
                for f in os.listdir(src):
                    src_f = os.path.join(src, f)
                    dst_f = os.path.join(dst, f)
                    if os.path.isfile(src_f) and f.endswith(".lst"):
                        # Rewrite paths: image column .png → .jpg
                        with open(src_f, "r") as fh:
                            lines = fh.readlines()
                        new_lines = []
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) == 2:
                                img_rel, lbl_rel = parts
                                # Convert image extension to .jpg
                                img_base = os.path.splitext(img_rel)[0]
                                new_lines.append(f"{img_base}.jpg {lbl_rel}\n")
                            else:
                                new_lines.append(line)
                        with open(dst_f, "w") as fh:
                            fh.writelines(new_lines)
                    elif os.path.isfile(src_f) and not os.path.exists(dst_f):
                        import shutil
                        shutil.copy2(src_f, dst_f)
                print(f"[RELLIS-3D] Copied and fixed {split_dir}/ → fast dir")

    # RUGD
    rugd_root = os.path.join(args.data_dir, "RUGD")
    if os.path.isdir(rugd_root):
        rugd_out = os.path.join(args.data_dir, "RUGD_fast")
        tasks = process_rugd(rugd_root, rugd_out)
        print(f"[RUGD] {len(tasks)} files to process → {rugd_out}")
        all_tasks.extend(tasks)

    # GOOSE
    goose_root = os.path.join(args.data_dir, "GOOSE")
    if os.path.isdir(goose_root):
        goose_out = os.path.join(args.data_dir, "GOOSE_fast")
        tasks = process_goose(goose_root, goose_out)
        print(f"[GOOSE] {len(tasks)} files to process → {goose_out}")
        all_tasks.extend(tasks)

        # Copy CSV mapping
        for root, dirs, files in os.walk(goose_root):
            for f in files:
                if f.endswith(".csv"):
                    src = os.path.join(root, f)
                    rel = os.path.relpath(src, goose_root)
                    dst = os.path.join(goose_out, rel)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    if not os.path.exists(dst):
                        import shutil
                        shutil.copy2(src, dst)

    if not all_tasks:
        print("No datasets found!")
        return

    print(f"\nTotal: {len(all_tasks)} image-label pairs")
    print(f"Max long side: {MAX_LONG_SIDE}px")
    print(f"Workers: {args.workers}")
    print(f"Processing...\n")

    t0 = time.time()
    done = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(resize_and_save, task): task
                   for task in all_tasks}

        for future in as_completed(futures):
            success, msg = future.result()
            if success:
                done += 1
            else:
                failed += 1
                if failed <= 5:
                    print(f"  [FAIL] {msg}")

            if done % 2000 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                remaining = (len(all_tasks) - done) / rate if rate > 0 else 0
                print(f"  [{done}/{len(all_tasks)}] "
                      f"{rate:.0f} img/s, ETA {remaining:.0f}s")

    elapsed = time.time() - t0
    print(f"\nDone! {done} processed, {failed} failed in {elapsed:.1f}s")
    print(f"Rate: {done/elapsed:.0f} images/sec")
    print(f"\nTo train with fast data:")
    print(f"  python scripts/train.py --fast")


if __name__ == "__main__":
    main()