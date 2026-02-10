#!/usr/bin/env python3
"""
Final environment and dataset verification script for:
- RELLIS-3D dataset
- Training environment
- Reproduction of Pickeral et al. (2024) settings

Run location: ~/offroad-segmentation/src/
"""

import os
import sys
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "Rellis-3D")
SPLIT_DIR = os.path.join(PROJECT_ROOT, "data", "Rellis-3D", "split")
EFFICIENTVIT_PATH = os.path.join(PROJECT_ROOT, "efficientvit")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

# Add project root and efficientvit to sys.path
sys.path.append(PROJECT_ROOT)
sys.path.insert(0, EFFICIENTVIT_PATH)

ORIG_TO_TRAIN = {
    0: 255, 1: 255,
    3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6,
    10: 7, 12: 8, 15: 9, 17: 10, 18: 11, 19: 12,
    23: 13, 27: 14, 31: 15, 33: 16, 34: 17,
}

errors = []
warnings = []

def ok(msg: str) -> None:
    print(f"  [OK] {msg}")

def fail(msg: str) -> None:
    print(f"  [ERR] {msg}")
    errors.append(msg)

def warn(msg: str) -> None:
    print(f"  [WARN] {msg}")
    warnings.append(msg)

print()
print("Paths")
print(f"  PROJECT_ROOT : {PROJECT_ROOT}")
print(f"  DATA_ROOT    : {DATA_ROOT}")
print(f"  SPLIT_DIR    : {SPLIT_DIR}")
print(f"  EFFICIENTVIT : {EFFICIENTVIT_PATH}")
print(f"  SRC_DIR      : {SRC_DIR}")
print()

# 1) GPU & PyTorch
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
        if "sm_120" in torch.cuda.get_arch_list():
            ok("Blackwell sm_120 support detected")
        else:
            fail("sm_120 not supported – need PyTorch cu128 with Blackwell support")
        x = torch.randn(100, 100).cuda()
        _ = torch.matmul(x, x)
        ok("Basic GPU compute test passed")
    else:
        fail("CUDA is not available")
except Exception as e:
    fail(f"PyTorch error: {e}")

# 2) Dataset directory
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
            ok(f"  {seq}/pylon_camera_node: {len(imgs)} images")
        else:
            fail(f"  {seq}/pylon_camera_node not found")

        if os.path.exists(lbl_dir):
            lbls = [f for f in os.listdir(lbl_dir) if f.endswith(".png")]
            total_labels += len(lbls)
            ok(f"  {seq}/pylon_camera_node_label_id: {len(lbls)} labels")
        else:
            fail(f"  {seq}/pylon_camera_node_label_id not found")

    ok(f"Total images: {total_images} / Total labels: {total_labels}")
    if total_images != total_labels:
        warn(f"Number of images ({total_images}) and labels ({total_labels}) differ "
             "(this is expected for RELLIS-3D: not all frames are labeled)")
else:
    fail(f"Data root not found: {DATA_ROOT}")

# 3) Split files
print()
print("=" * 60)
print("[3/7] Split files")
print("=" * 60)

split_counts = {}
for split_name in ["train.lst", "val.lst", "test.lst"]:
    split_path = os.path.join(SPLIT_DIR, split_name)
    if os.path.exists(split_path):
        with open(split_path, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        split_counts[split_name] = len(lines)
        ok(f"{split_name}: {len(lines)} entries")

        missing_img = 0
        missing_lbl = 0
        checked = min(len(lines), 20)
        for line in lines[:checked]:
            parts = line.split()
            if len(parts) >= 2:
                if not os.path.exists(os.path.join(DATA_ROOT, parts[0])):
                    missing_img += 1
                if not os.path.exists(os.path.join(DATA_ROOT, parts[1])):
                    missing_lbl += 1
        if missing_img == 0 and missing_lbl == 0:
            ok(f"  Path check passed (sample {checked} entries)")
        else:
            fail(f"  Path mismatch: {missing_img} images and {missing_lbl} labels missing")

        print(f"    Example: {lines[0][:80]}...")
    else:
        fail(f"{split_name} not found: {split_path}")

if "train.lst" in split_counts:
    total_split = sum(split_counts.values())
    train_ratio = split_counts["train.lst"] / total_split * 100
    ok(f"Train ratio: {train_ratio:.1f}% (reference in paper: 70%)")
    if abs(train_ratio - 70) > 10:
        warn(f"Train/test ratio differs from 70/30 used in the paper: {train_ratio:.1f}%")

# 4) Label values
print()
print("=" * 60)
print("[4/7] Label values (sample)")
print("=" * 60)
try:
    train_lst = os.path.join(SPLIT_DIR, "train.lst")
    with open(train_lst, "r") as f:
        first_line = f.readline().strip().split()

    lbl_path = os.path.join(DATA_ROOT, first_line[1])
    lbl = np.array(Image.open(lbl_path))
    unique_ids = sorted(np.unique(lbl).tolist())
    ok(f"Label shape: {lbl.shape}, dtype: {lbl.dtype}")
    ok(f"Unique label IDs: {unique_ids}")

    known_ids = set(ORIG_TO_TRAIN.keys())
    unknown = [v for v in unique_ids if v not in known_ids]
    if not unknown:
        ok("All label IDs are covered by the mapping table")
    else:
        warn(f"Label IDs not in mapping table: {unknown} (will be ignored in training)")

    img_path = os.path.join(DATA_ROOT, first_line[0])
    img = Image.open(img_path).convert("RGB")
    img_arr = np.array(img)
    ok(f"Image shape: {img_arr.shape} ({img_arr.dtype}), ext: .{img_path.split('.')[-1]}")
    ok(f"Image resolution: {img.size[0]}x{img.size[1]} (paper: 1920x1200)")
except Exception as e:
    fail(f"Label check failed: {e}")

# 5) EfficientViT load + head replace
print()
print("=" * 60)
print("[5/7] EfficientViT load and head replacement")
print("=" * 60)
try:
    sys.path.insert(0, EFFICIENTVIT_PATH)
    from efficientvit.seg_model_zoo import create_efficientvit_seg_model
    import torch
    import torch.nn as nn

    model = create_efficientvit_seg_model("efficientvit-seg-b0-cityscapes", pretrained=True)
    ok("EfficientViT-Seg-B0 (Cityscapes pretrained) loaded")

    params_before = sum(p.numel() for p in model.parameters())
    ok(f"Parameters (before head replace): {params_before:,} ({params_before/1e6:.2f}M)")

    replaced = False
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.out_channels == 19:
            parts = name.rsplit(".", 1)
            parent = dict(model.named_modules())[parts[0]]
            new_conv = nn.Conv2d(
                module.in_channels,
                18,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                bias=module.bias is not None,
            )
            setattr(parent, parts[1], new_conv)
            ok(f"Head replaced: {name} (19 → 18 classes)")
            replaced = True
            break
    if not replaced:
        fail("Failed to find Conv2d head with out_channels=19")

    params_after = sum(p.numel() for p in model.parameters())
    ok(f"Parameters (after head replace): {params_after:,} ({params_after/1e6:.2f}M)")
except Exception as e:
    fail(f"EfficientViT load failed: {e}")

# 6) Forward pass + speed
print()
print("=" * 60)
print("[6/7] Forward pass and inference speed (GPU)")
print("=" * 60)
try:
    import time
    model = model.cuda().eval()
    dummy = torch.randn(1, 3, 512, 512).cuda()

    with torch.no_grad():
        output = model(dummy)
    ok(f"Input: {list(dummy.shape)} → Output: {list(output.shape)}")

    if output.shape[1] == 18:
        ok(f"Output classes: {output.shape[1]} (expected: 18)")
    else:
        fail(f"Output class count mismatch: {output.shape[1]} != 18")

    torch.cuda.synchronize()
    times = []
    for _ in range(100):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            _ = model(dummy)
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    avg_ms = np.mean(times[20:]) * 1000
    fps = 1000 / avg_ms
    ok(f"Inference speed: {avg_ms:.2f} ms ({fps:.1f} FPS) @ 512x512")
    ok("Reference in paper (V100): 11.53 ms")

    del model, dummy
    torch.cuda.empty_cache()
except Exception as e:
    fail(f"Forward pass failed: {e}")

# 7) Paper reproduction checklist
print()
print("=" * 60)
print("[7/7] Paper reproduction checklist")
print("        (Pickeral et al., 2024 - SCIRP)")
print("=" * 60)

paper_checks = {
    "Dataset":     "RELLIS-3D (6,234 labeled RGB)",
    "Classes":     "18 (dirt excluded, void=ignore)",
    "Model":       "EfficientViT-Seg-B0 (pretrained)",
    "Optimizer":   "AdamW (lr=0.001)",
    "Scheduler":   "Cosine Annealing + 20ep warm-up",
    "Augmentation":"Flip/Crop/Hue/Erasing",
    "Loss":        "CrossEntropyLoss(ignore=255)",
    "Metric":      "mIoU (mean IoU over classes)",
    "Target mIoU": "~76.03%",
}

train_py_path = os.path.join(SRC_DIR, "train.py")
train_code = ""
if os.path.exists(train_py_path):
    with open(train_py_path, "r") as f:
        train_code = f.read()
    ok(f"train.py found: {train_py_path}")
else:
    warn(f"train.py not found: {train_py_path}")

dataset_py_path = os.path.join(SRC_DIR, "dataset.py")
dataset_code = ""
if os.path.exists(dataset_py_path):
    with open(dataset_py_path, "r") as f:
        dataset_code = f.read()
    ok(f"dataset.py found: {dataset_py_path}")
else:
    warn(f"dataset.py not found: {dataset_py_path}")

if train_code:
    if "AdamW" in train_code:
        ok("Optimizer: AdamW detected")
    else:
        warn("AdamW not found in train.py")

    if "0.001" in train_code:
        ok("Base LR: 0.001 detected")
    else:
        warn("Base LR 0.001 not found in train.py")

    if "20" in train_code and ("warmup" in train_code.lower() or "LinearLR" in train_code):
        ok("Warm-up: 20 epochs detected")
    else:
        warn("20 epoch warm-up not clearly detected in train.py")

    if "CosineAnnealing" in train_code:
        ok("Scheduler: CosineAnnealingLR detected")
    else:
        warn("CosineAnnealingLR not found in train.py")

    if "CrossEntropyLoss" in train_code and "255" in train_code:
        ok("Loss: CrossEntropyLoss(ignore_index=255) detected")
    else:
        warn("CrossEntropyLoss(ignore_index=255) not found in train.py")

    if "18" in train_code or "NUM_CLASSES" in train_code:
        ok("Number of classes: 18 detected")
    else:
        warn("18-class setting not clearly detected in train.py")

if dataset_code:
    aug_checks = {
        "Random Flip":    "hflip" in dataset_code,
        "Random Crop":    "RandomCrop" in dataset_code,
        "Hue Change":     "adjust_hue" in dataset_code,
        "Random Erasing": "RandomErasing" in dataset_code,
    }
    for aug_name, found in aug_checks.items():
        if found:
            ok(f"Augmentation: {aug_name} detected")
        else:
            warn(f"Augmentation {aug_name} not found in dataset.py")

print()
print("Paper training settings summary:")
print("  Key         | Value")
print(" -------------+------------------------------------------")
for k, v in paper_checks.items():
    print(f"  {k:11s} | {v}")
print("")

# Final summary
print("=" * 60)
print(f"Final result: {len(errors)} errors, {len(warnings)} warnings")
print("=" * 60)

if errors:
    print("\n[Errors - must be fixed]")
    for e in errors:
        print(f"  - {e}")

if warnings:
    print("\n[Warnings - recommended to check]")
    for w in warnings:
        print(f"  - {w}")

if not errors:
    print("\nEnvironment is ready. You can start training with:")
    print("  cd ~/offroad-segmentation")
    print("  conda activate offroad")
    print("  python src/train.py")
else:
    print("\nPlease fix the errors above and run verification again.")
