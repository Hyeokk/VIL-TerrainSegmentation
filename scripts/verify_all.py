#!/usr/bin/env python3
"""
Environment and dataset verification for the unified 7-class off-road
segmentation project with dual-model (EfficientViT + FFNet) support.

Verifies:
  1. GPU & PyTorch (Blackwell sm_120)
  2. RELLIS-3D dataset directory
  3. Split files
  4. Label values (7-class unified ontology)
  5. Model loading (EfficientViT + FFNet)
  6. Forward pass + speed benchmark
  7. Training configuration checklist

Usage:
    conda activate offroad
    python scripts/verify_all.py
"""

import os
import sys
import time
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "Rellis-3D")
SPLIT_DIR = os.path.join(PROJECT_ROOT, "data", "Rellis-3D", "split")
EFFICIENTVIT_PATH = os.path.join(PROJECT_ROOT, "efficientvit")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

sys.path.append(PROJECT_ROOT)
sys.path.insert(0, EFFICIENTVIT_PATH)

# 7-class unified ontology mapping (from src/dataset.py)
from src.dataset import RELLIS_TO_UNIFIED, NUM_CLASSES, CLASS_NAMES

errors = []
warnings = []

def ok(msg):
    print(f"  [OK]   {msg}")

def fail(msg):
    print(f"  [ERR]  {msg}")
    errors.append(msg)

def warn(msg):
    print(f"  [WARN] {msg}")
    warnings.append(msg)

print()
print("Paths")
print(f"  PROJECT_ROOT : {PROJECT_ROOT}")
print(f"  DATA_ROOT    : {DATA_ROOT}")
print(f"  EFFICIENTVIT : {EFFICIENTVIT_PATH}")
print()

# ======================================================================
# 1) GPU & PyTorch
# ======================================================================
print("=" * 60)
print("[1/7] GPU & PyTorch")
print("=" * 60)
try:
    import torch
    import torchvision

    ok(f"PyTorch {torch.__version__}")
    ok(f"Torchvision {torchvision.__version__}")
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        ok(f"GPU: {gpu} (CC {cc[0]}.{cc[1]}, {mem:.1f} GB)")
        arch_list = torch.cuda.get_arch_list()
        if "sm_120" in arch_list:
            ok("Blackwell sm_120 support detected")
        else:
            warn(f"sm_120 not in arch list {arch_list} (may work on non-Blackwell GPU)")
        x = torch.randn(100, 100).cuda()
        _ = torch.matmul(x, x)
        ok("Basic GPU compute test passed")
    else:
        fail("CUDA is not available")
except Exception as e:
    fail(f"PyTorch error: {e}")

# ======================================================================
# 2) Dataset directory
# ======================================================================
print()
print("=" * 60)
print("[2/7] RELLIS-3D directory")
print("=" * 60)

total_images = 0
total_labels = 0
if os.path.exists(DATA_ROOT):
    ok(f"Data root exists: {DATA_ROOT}")
    sequences = sorted(
        d for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d)) and d.startswith("0000")
    )
    if sequences:
        ok(f"Sequence folders: {sequences}")
    else:
        fail("No sequence folders (00000~00004) found")

    for seq in sequences:
        img_dir = os.path.join(DATA_ROOT, seq, "pylon_camera_node")
        lbl_dir = os.path.join(DATA_ROOT, seq, "pylon_camera_node_label_id")
        if os.path.exists(img_dir):
            imgs = [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))]
            total_images += len(imgs)
            ok(f"  {seq}/images: {len(imgs)}")
        else:
            fail(f"  {seq}/pylon_camera_node not found")
        if os.path.exists(lbl_dir):
            lbls = [f for f in os.listdir(lbl_dir) if f.endswith(".png")]
            total_labels += len(lbls)
            ok(f"  {seq}/labels: {len(lbls)}")
        else:
            fail(f"  {seq}/pylon_camera_node_label_id not found")

    ok(f"Total: {total_images} images / {total_labels} labels")
else:
    fail(f"Data root not found: {DATA_ROOT}")

# ======================================================================
# 3) Split files
# ======================================================================
print()
print("=" * 60)
print("[3/7] Split files")
print("=" * 60)

for split_name in ["train.lst", "val.lst", "test.lst"]:
    split_path = os.path.join(SPLIT_DIR, split_name)
    if os.path.exists(split_path):
        with open(split_path, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        ok(f"{split_name}: {len(lines)} entries")
    else:
        warn(f"{split_name} not found: {split_path}")

custom_dir = os.path.join(DATA_ROOT, "split_custom")
for split_name in ["train_70.lst", "test_30.lst"]:
    split_path = os.path.join(custom_dir, split_name)
    if os.path.exists(split_path):
        with open(split_path, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        ok(f"custom/{split_name}: {len(lines)} entries")
    else:
        warn(f"custom/{split_name} not found (run: python scripts/make_split_custom.py)")

# ======================================================================
# 4) Label values (7-class unified)
# ======================================================================
print()
print("=" * 60)
print("[4/7] Label values — 7-class unified ontology")
print("=" * 60)
try:
    train_lst = os.path.join(SPLIT_DIR, "train.lst")
    if os.path.exists(train_lst):
        with open(train_lst, "r") as f:
            first_line = f.readline().strip().split()
        lbl_path = os.path.join(DATA_ROOT, first_line[1])
        lbl = np.array(Image.open(lbl_path))
        unique_ids = sorted(np.unique(lbl).tolist())
        ok(f"Label shape: {lbl.shape}, dtype: {lbl.dtype}")
        ok(f"Unique original IDs: {unique_ids}")

        known_ids = set(RELLIS_TO_UNIFIED.keys())
        unknown = [v for v in unique_ids if v not in known_ids]
        if not unknown:
            ok("All label IDs covered by RELLIS_TO_UNIFIED mapping")
        else:
            warn(f"Unknown IDs (will be ignored=255): {unknown}")

        # Verify remapping produces valid 7-class output
        remapped = np.full_like(lbl, 255, dtype=np.int64)
        for orig_id, unified_id in RELLIS_TO_UNIFIED.items():
            remapped[lbl == orig_id] = unified_id
        unified_ids = sorted(np.unique(remapped).tolist())
        ok(f"Remapped unified IDs: {unified_ids}")
        valid_ids = set(range(NUM_CLASSES)) | {255}
        invalid = [v for v in unified_ids if v not in valid_ids]
        if not invalid:
            ok(f"All remapped IDs valid for {NUM_CLASSES}-class ontology")
        else:
            fail(f"Invalid remapped IDs: {invalid}")

        ok(f"Ontology: {NUM_CLASSES} classes = {CLASS_NAMES}")
    else:
        warn("Cannot check labels — train.lst not found")
except Exception as e:
    fail(f"Label check failed: {e}")

# ======================================================================
# 5) Model loading
# ======================================================================
print()
print("=" * 60)
print("[5/7] Model loading — EfficientViT + FFNet")
print("=" * 60)

model_for_speed = None

try:
    from src.models import build_model, SUPPORTED_MODELS

    ok(f"models.py loaded, {len(SUPPORTED_MODELS)} models registered")

    # Test EfficientViT-B0 (always available)
    try:
        model = build_model("efficientvit-b0", num_classes=NUM_CLASSES, pretrained=True)
        params = sum(p.numel() for p in model.parameters())
        ok(f"efficientvit-b0: {params:,} params ({params/1e6:.2f}M)")
        model_for_speed = model
    except Exception as e:
        fail(f"efficientvit-b0 load failed: {e}")

    # Test EfficientViT-B1
    try:
        model_b1 = build_model("efficientvit-b1", num_classes=NUM_CLASSES, pretrained=True)
        params = sum(p.numel() for p in model_b1.parameters())
        ok(f"efficientvit-b1: {params:,} params ({params/1e6:.2f}M)")
        del model_b1
    except Exception as e:
        warn(f"efficientvit-b1 load failed (may need download): {e}")

    # Test FFNet
    ffnet_available = False
    try:
        model_ff = build_model("ffnet-78s", num_classes=NUM_CLASSES, pretrained=True)
        params = sum(p.numel() for p in model_ff.parameters())
        ok(f"ffnet-78s: {params:,} params ({params/1e6:.2f}M)")
        ffnet_available = True
        del model_ff
    except Exception as e:
        warn(f"ffnet-78s load failed: {e}")
        warn("FFNet not available. To install:")
        warn("  git clone https://github.com/Qualcomm-AI-research/FFNet.git")
        warn("  OR pip install qai-hub-models")

except Exception as e:
    fail(f"Model factory import failed: {e}")

# ======================================================================
# 6) Forward pass + speed
# ======================================================================
print()
print("=" * 60)
print("[6/7] Forward pass and inference speed")
print("=" * 60)

if model_for_speed is not None:
    try:
        model_for_speed = model_for_speed.cuda().eval()
        dummy = torch.randn(1, 3, 512, 512).cuda()

        with torch.no_grad():
            output = model_for_speed(dummy)
        ok(f"Input: {list(dummy.shape)} → Output: {list(output.shape)}")

        if output.shape[1] == NUM_CLASSES:
            ok(f"Output classes: {output.shape[1]} (expected: {NUM_CLASSES})")
        else:
            fail(f"Output classes mismatch: {output.shape[1]} != {NUM_CLASSES}")

        torch.cuda.synchronize()
        times = []
        for _ in range(100):
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                _ = model_for_speed(dummy)
            torch.cuda.synchronize()
            times.append(time.time() - t0)
        avg_ms = np.mean(times[20:]) * 1000
        fps = 1000 / avg_ms
        ok(f"Inference speed: {avg_ms:.2f} ms ({fps:.1f} FPS) @ 512x512")

        del model_for_speed, dummy
        torch.cuda.empty_cache()
    except Exception as e:
        fail(f"Forward pass failed: {e}")
else:
    warn("Skipping speed test — no model loaded")

# ======================================================================
# 7) Training config checklist
# ======================================================================
print()
print("=" * 60)
print("[7/7] Training configuration checklist")
print("=" * 60)

checklist = {
    "Ontology":       f"Unified {NUM_CLASSES}-class (caterpillar-aware)",
    "Datasets":       "RELLIS-3D + RUGD + GOOSE (auto-detected)",
    "Models":         "EfficientViT-B0/B1, FFNet-40S/54S/78S",
    "Optimizer":      "AdamW (lr=0.001, wd=0.01)",
    "Scheduler":      "LinearLR warmup 20ep + CosineAnnealing",
    "Loss":           "Focal Loss (gamma=2.0, per-class weights)",
    "EMA":            "decay=0.9999",
    "Augmentation":   "Flip, MultiScaleCrop, ColorJitter, GaussBlur, Shadow, Erasing",
    "Grad Clipping":  "max_norm=5.0",
    "Target deploy":  "Qualcomm IQ-9075 (100 TOPS NPU, INT8)",
}

print()
for k, v in checklist.items():
    print(f"  {k:16s} │ {v}")
print()

# ======================================================================
# Summary
# ======================================================================
print("=" * 60)
print(f"Result: {len(errors)} errors, {len(warnings)} warnings")
print("=" * 60)

if errors:
    print("\n[Errors — must fix]")
    for e in errors:
        print(f"  ✗ {e}")

if warnings:
    print("\n[Warnings — check]")
    for w in warnings:
        print(f"  △ {w}")

if not errors:
    print("\nEnvironment ready. Start training with:")
    print("  python scripts/train.py --model efficientvit-b1")
    print("  python scripts/train.py --model ffnet-78s")
else:
    print("\nPlease fix errors and re-run verification.")