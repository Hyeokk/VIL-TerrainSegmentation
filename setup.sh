#!/bin/bash
# ============================================================
# Off-Road Segmentation Environment Setup Script
# GPU   : NVIDIA RTX PRO 4000 Blackwell (sm_120)
# Driver: nvidia-580-open (CUDA 13.0 compatible)
# ============================================================

set -e  # exit immediately on error

ENV_NAME=offroad

echo "=========================================="
echo " [1/6] Create Conda environment"
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
echo " [2/6] Install PyTorch (cu128 - Blackwell sm_120 required)"
echo "=========================================="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128


echo "=========================================="
echo " [3/6] Verify PyTorch GPU (Blackwell)"
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
echo " [4/6] Install Python dependencies"
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
echo " [5/6] Install DDRNet23-Slim (Qualcomm AI Hub Models)"
echo "=========================================="

# Qualcomm AI Hub Models - provides DDRNet23-Slim with pretrained weights
# and Qualcomm NPU-verified architecture for IQ-9075 deployment
pip install qai-hub-models

# Verify DDRNet23-Slim is available
python -c "
from qai_hub_models.models.ddrnet23_slim import Model
print('>>> DDRNet23-Slim (qai_hub_models) OK!')
"

# Optional: Qualcomm AI Hub CLI for cloud compile + profiling
# Requires API token from https://aihub.qualcomm.com
# Uncomment if needed:
# pip install qai-hub


echo "=========================================="
echo " [6/6] Clone EfficientViT source (legacy model support)"
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
echo "  1) Download and unpack datasets into data/:"
echo "     - RELLIS-3D (required): data/Rellis-3D/"
echo "     - RUGD (optional):     data/RUGD/"
echo "     - GOOSE (optional):    data/GOOSE/"
echo "     See data/README.md for download links."
echo ""
echo "  2) Create custom 70/30 train/test split:"
echo "     conda activate ${ENV_NAME}"
echo "     python scripts/make_split_custom.py"
echo ""
echo "  3) Verify environment:"
echo "     conda activate ${ENV_NAME}"
echo "     python scripts/verify_all.py"
echo ""
echo "  4) (Recommended) Preprocess for fast training:"
echo "     python scripts/preprocess_datasets.py"
echo ""
echo "  5) Train DDRNet23-Slim (recommended for IQ-9075):"
echo "     python scripts/train.py --model ddrnet23-slim --fast --num_workers 8"
echo ""
echo "     Or train EfficientViT-B1 (legacy, needs assets/ weights):"
echo "     python scripts/train.py --model efficientvit-b1 --fast --num_workers 8"
echo ""