import os
import sys
sys.path.insert(0, './efficientvit'); sys.path.append('.')  # EfficientViT source path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.dataset import Rellis3DDataset, NUM_CLASSES
from efficientvit.seg_model_zoo import create_efficientvit_seg_model


def main():
    # Create checkpoints directory if it doesn't exist
    os.makedirs("./checkpoints", exist_ok=True)

    # === Dataset ===
    train_set = Rellis3DDataset(
        data_root="./data/Rellis-3D",
        split_file="./data/Rellis-3D/split_custom/train_70.lst",
        is_train=True,
        crop_size=512,
    )
    val_set = Rellis3DDataset(
        data_root="./data/Rellis-3D",
        split_file="./data/Rellis-3D/split_custom/test_30.lst",
        is_train=False,
        crop_size=512,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=4,
        shuffle=False,
        num_workers=4,
    )

    # === Model ===
    model = create_efficientvit_seg_model(
        "efficientvit-seg-b0-cityscapes",
        pretrained=True,
    )

    # Replace head: 19 (Cityscapes) → 18 (RELLIS-3D)
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.out_channels == 19:
            parent = dict(model.named_modules())[name.rsplit(".", 1)[0]]
            attr = name.rsplit(".", 1)[1]
            new_conv = nn.Conv2d(
                module.in_channels,
                NUM_CLASSES,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                bias=(module.bias is not None),
            )
            setattr(parent, attr, new_conv)
            print(f"Replaced {name}: out_channels 19 → {NUM_CLASSES}")
            break

    model = model.cuda()

    # === Loss / Optimizer / Scheduler ===
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    warmup = LinearLR(optimizer, start_factor=1e-6, total_iters=20)
    cosine = CosineAnnealingLR(optimizer, T_max=180)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[20])

    # === Training loop ===
    best_miou = 0.0  # placeholder if you later add evaluation
    for epoch in range(200):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            outputs = model(images)

            if outputs.shape[2:] != labels.shape[1:]:
                outputs = nn.functional.interpolate(
                    outputs,
                    size=labels.shape[1:],
                    mode="bilinear",
                    align_corners=False,
                )

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/200 | LR: {current_lr:.6f} | "
            f"Loss: {avg_loss:.4f}"
        )

        # Save every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(
                model.state_dict(),
                f"./checkpoints/epoch_{epoch + 1}.pth",
            )

    torch.save(model.state_dict(), "./checkpoints/final_model.pth")


if __name__ == "__main__":
    main()
