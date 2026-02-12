#!/bin/bash
# ============================================================
# Off-Road Segmentation Environment Setup Script
# GPU   : NVIDIA RTX PRO 4000 Blackwell (sm_120)
# Driver: nvidia-580-open (CUDA 13.0 compatible)
# ============================================================

set -e  # exit immediately on error

ENV_NAME=offroad

echo "=========================================="
echo " [1/5] Create Conda environment"
echo "=========================================="

# Create env only if it does not exist
if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda create -n ${ENV_NAME} python=3.11 -y
else
    echo "Conda env '${ENV_NAME}' already exists. Skipping creation."
fi

# Activate env inside bash script
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}


echo "=========================================="
echo " [2/5] Install PyTorch (cu128 - Blackwell sm_120 required)"
echo "=========================================="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128


echo "=========================================="
echo " [3/5] Verify PyTorch GPU (Blackwell)"
echo "=========================================="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Compute Capability: {torch.cuda.get_device_capability(0)}')
    arch_list = torch.cuda.get_arch_list()
    print(f'Supported Archs: {arch_list}')
    assert 'sm_120' in arch_list, 'ERROR: sm_120 not found! Blackwell not supported.'
    print('>>> Blackwell (sm_120) OK!')
    x = torch.randn(100, 100).cuda()
    print(f'>>> GPU Tensor Test: {x.shape} on {x.device} - OK!')
else:
    raise SystemExit('ERROR: CUDA is not available.')
"


echo "=========================================="
echo " [4/5] Install Python dependencies"
echo "=========================================="

# Core libs and EfficientViT dependencies
pip install \
    numpy \
    pillow \
    tqdm \
    timm \
    einops \
    opencv-python \
    scipy \
    matplotlib \
    torchmetrics \
    torchprofile

# ONNX export and tooling
pip install \
    onnx \
    onnxscript \
    onnxsim \
    onnxruntime

# Hydra / config libs often used by EfficientViT
pip install \
    omegaconf \
    hydra-core \
    yacs

# Optional: logging / tracking
pip install wandb

# Optional: Segment Anything dependency (some EfficientViT modules may import it)
pip install git+https://github.com/facebookresearch/segment-anything.git


echo "=========================================="
echo " [5/5] Clone EfficientViT source"
echo "=========================================="
if [ ! -d "efficientvit" ]; then
    git clone https://github.com/mit-han-lab/efficientvit.git
    echo ">>> EfficientViT cloned."
else
    echo ">>> EfficientViT directory already exists. Skipping clone."
fi


# Directories will be created by python scripts as needed

echo ""
echo "============================================"
echo " Environment setup completed!"
echo "============================================"
echo ""
echo " Next steps:"
echo "  1) Download and unpack RELLIS-3D dataset into data/Rellis-3D/:"
echo "     - Full images       → data/Rellis-3D/0000x/pylon_camera_node/"
echo "     - ID annotations    → data/Rellis-3D/0000x/pylon_camera_node_label_id/"
echo "     - Split files       → data/Rellis-3D/split/train.lst, val.lst, test.lst"
echo ""
echo "  2) (Optional) Create custom 70/30 train/test split:"
echo "     conda activate ${ENV_NAME}"
echo "     python scripts/make_split_custom.py"
echo ""
echo "  3) Verify environment:"
echo "     conda activate ${ENV_NAME}"
echo "     python scripts/verify_all.py"
echo ""
echo "  4) Start training:"
echo "     conda activate ${ENV_NAME}"
echo "     python scripts/train.py --model efficientvit-b1"
echo ""