# Off-Road Semantic Segmentation on RELLIS-3D with EfficientViT

This repository contains the pipeline for fine-tuning **EfficientViT-Seg** on the **RELLIS-3D** dataset for off-road semantic segmentation.

## Methodology

We fine-tune the EfficientViT model following the methodology described in:
> **"Efficient Vision Transformers for Autonomous Off-Road Perception Systems"** (Pickeral et al., 2024)
> [https://doi.org/10.4236/jcc.2024.129011](https://doi.org/10.4236/jcc.2024.129011)

The training hyperparameters are set to match those specified in the paper to reproduce the results.

### Backbone Model
We use the pre-trained **EfficientViT-Seg-B0** model trained on Cityscapes as the backbone:
- [efficientvit_seg_b0_cityscapes.pt](https://huggingface.co/han-cai/efficientvit-seg/resolve/main/efficientvit_seg_b0_cityscapes.pt)

## Dataset

We use the **RELLIS-3D** dataset, a multi-modal dataset for off-road robotics.
- **Source**: [https://github.com/unmannedlab/RELLIS-3D/tree/main](https://github.com/unmannedlab/RELLIS-3D/tree/main)
- **Data Preparation**: Download the full images and annotations from the repository.

### Directory Structure
Ensure your data is organized as follows:
```
data/
└── Rellis-3D/
    ├── 00000/ ... 00004/           # Raw images & labels
    ├── split/
    │   ├── train.lst
    │   ├── val.lst
    │   └── test.lst
    └── split_custom/
        ├── train_70.lst            # Custom 70% split
        └── test_30.lst             # Custom 30% split
```

## Setup

### 1. Clone and Install
Clone the repository into a directory named `efficientvit_rellis` and run the setup script:

```bash
# Clone the repository
git clone https://github.com/hyeokk/VIL-TerrainSegmentation.git efficientvit_rellis

# Enter the directory
cd efficientvit_rellis

# Setup environment (conda, pytorch, dependencies)
# Make sure to initialize conda first (e.g., source ~/anaconda3/etc/profile.d/conda.sh)
bash setup.sh
```

## Usage

### 1. Verify Environment
Check if all dependencies and the GPU are correctly configured:
```bash
conda activate offroad
python scripts/verify_all.py
```

### 2. Prepare Data Split (Optional)
Create a custom 70/30 train/test split if needed:
```bash
python scripts/make_split_custom.py
```

### 3. Training
Run the fine-tuning script. The script automatically handles:
- Loading the Cityscapes pre-trained weights.
- Replacing the segmentation head (19 classes -> RELLIS-3D classes).
- Setting up the optimizer (AdamW) and scheduler (CosineAnnealingLR) with warmup.

```bash
conda activate offroad
python scripts/train.py
```

Checkpoints will be saved in the `checkpoints/` directory.

## References
- **EfficientViT**: [https://github.com/mit-han-lab/efficientvit](https://github.com/mit-han-lab/efficientvit)
- **RELLIS-3D**: [https://github.com/unmannedlab/RELLIS-3D](https://github.com/unmannedlab/RELLIS-3D)

## License
This project is based on open-source code and datasets:
- **EfficientViT** code is licensed under the **Apache License 2.0**.
- **RELLIS-3D** dataset is licensed under **Creative Commons Attribution-NonCommercial-ShareAlike 3.0 (CC BY-NC-SA 3.0)**.

Please ensure you comply with the respective licenses when using this code and dataset, especially the **Non-Commercial** usage restriction of RELLIS-3D.

## Acknowledgements
- Thanks to the authors of EfficientViT for their efficient architecture.
- Thanks to the UNMANNED Lab for providing the RELLIS-3D dataset.
