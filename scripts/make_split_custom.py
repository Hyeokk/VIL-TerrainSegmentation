#!/usr/bin/env python3
"""
Create custom 70/30 train/test split for RELLIS-3D.

Input  : data/Rellis-3D/split/train.lst, val.lst, test.lst
Output : data/Rellis-3D/split_custom/train_70.lst, test_30.lst
"""

import os
import random

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPLIT_DIR = os.path.join(PROJECT_ROOT, "data", "Rellis-3D", "split")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "Rellis-3D", "split_custom")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    all_lines = []
    for name in ["train.lst", "val.lst", "test.lst"]:
        path = os.path.join(SPLIT_DIR, name)
        if not os.path.exists(path):
            print(f"[WARN] Split file not found: {path}")
            continue
        with open(path, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        print(f"[INFO] Loaded {len(lines)} entries from {name}")
        all_lines.extend(lines)

    total = len(all_lines)
    if total == 0:
        print("[ERROR] No entries found in split files.")
        return

    print(f"[INFO] Total labeled images: {total}")
    random.seed(42)
    random.shuffle(all_lines)

    split_idx = int(total * 0.7)
    train_lines = all_lines[:split_idx]
    test_lines = all_lines[split_idx:]

    train_path = os.path.join(OUT_DIR, "train_70.lst")
    test_path = os.path.join(OUT_DIR, "test_30.lst")

    with open(train_path, "w") as f:
        f.write("\n".join(train_lines) + "\n")
    with open(test_path, "w") as f:
        f.write("\n".join(test_lines) + "\n")

    print(f"[INFO] Saved train split  (70%): {len(train_lines)} → {train_path}")
    print(f"[INFO] Saved test split   (30%): {len(test_lines)} → {test_path}")
    print("[DONE] Custom split created.")

if __name__ == "__main__":
    main()
