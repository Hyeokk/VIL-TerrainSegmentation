import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import random

# Original ID → train ID mapping (void, dirt → ignore 255)
ORIG_TO_TRAIN = {
    0: 255, 1: 255,       # void, dirt → ignore
    3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6,
    10: 7, 12: 8, 15: 9, 17: 10, 18: 11, 19: 12,
    23: 13, 27: 14, 31: 15, 33: 16, 34: 17,
}
NUM_CLASSES = 18


class Rellis3DDataset(Dataset):
    def __init__(self, data_root, split_file, is_train=True, crop_size=512):
        """
        Args:
            data_root:  "./data/Rellis-3D"
            split_file: "./data/Rellis-3D/split/train.lst" (or custom split)
        """
        self.data_root = data_root
        self.is_train = is_train
        self.crop_size = crop_size

        # Load image/label pairs from split file
        self.pairs = []
        with open(split_file, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 2:
                    img_path = os.path.join(data_root, parts[0])
                    lbl_path = os.path.join(data_root, parts[1])
                    if os.path.exists(img_path) and os.path.exists(lbl_path):
                        self.pairs.append((img_path, lbl_path))

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.random_erasing = transforms.RandomErasing(p=0.5, scale=(0.02, 0.33))

        print(f"Loaded {len(self.pairs)} image-label pairs from {split_file}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, lbl_path = self.pairs[idx]

        image = Image.open(img_path).convert("RGB")
        mask = np.array(Image.open(lbl_path))

        # Original ID → train ID
        label = np.full_like(mask, 255, dtype=np.int64)
        for orig_id, train_id in ORIG_TO_TRAIN.items():
            label[mask == orig_id] = train_id
        label = Image.fromarray(label.astype(np.uint8))

        if self.is_train:
            # RELLIS-3D original resolution is 1920x1200 (w x h).
            # We apply augmentations directly on the original size.

            # 1) Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                label = TF.hflip(label)

            # Ensure image is at least crop_size
            w, h = image.size  # (width, height)
            if h < self.crop_size or w < self.crop_size:
                new_h = max(h, self.crop_size)
                new_w = max(w, self.crop_size)
                image = image.resize((new_w, new_h), Image.BILINEAR)
                label = label.resize((new_w, new_h), Image.NEAREST)
                w, h = new_w, new_h

            # 2) Random crop (crop_size x crop_size)
            i, j, th, tw = transforms.RandomCrop.get_params(
                image, output_size=(self.crop_size, self.crop_size)
            )
            image = TF.crop(image, i, j, th, tw)
            label = TF.crop(label, i, j, th, tw)

            # 3) Hue changing
            image = TF.adjust_hue(image, random.uniform(-0.1, 0.1))

        else:
            # Validation / test:
            # Center crop (or resize up if needed, then center crop)
            w, h = image.size
            if h < self.crop_size or w < self.crop_size:
                new_h = max(h, self.crop_size)
                new_w = max(w, self.crop_size)
                image = image.resize((new_w, new_h), Image.BILINEAR)
                label = label.resize((new_w, new_h), Image.NEAREST)
                w, h = new_w, new_h

            th, tw = self.crop_size, self.crop_size
            i = max(0, (h - th) // 2)
            j = max(0, (w - tw) // 2)
            image = TF.crop(image, i, j, th, tw)
            label = TF.crop(label, i, j, th, tw)

        # To tensor + normalize
        image = TF.to_tensor(image)
        image = self.normalize(image)
        label = torch.tensor(np.array(label), dtype=torch.long)

        # 4) Random erasing (train only)
        if self.is_train:
            image = self.random_erasing(image)

        return image, label
