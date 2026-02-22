# Pretrained Weights

This directory contains pretrained model weights used for fine-tuning. These files are not included in the repository and must be downloaded manually.

---

## DDRNet23-Slim (Recommended)

| Item | Value |
|------|-------|
| Model | DDRNet23-Slim Segmentation |
| Pretrained on | Cityscapes (19 classes) |
| Parameters | 5.7M |
| Cityscapes mIoU | 77.8% |
| Source | [ydhongHIT/DDRNet](https://github.com/ydhongHIT/DDRNet) |
| Provider | [Qualcomm AI Hub Models](https://github.com/qualcomm/ai-hub-models) |
| License | MIT (model), BSD-3 (qai_hub_models) |

### Download

**DDRNet23-Slim weights are auto-downloaded** by `qai_hub_models`. No manual download is needed.

When you first run training, `Model.from_pretrained()` will automatically download the Cityscapes pretrained weights to the pip cache directory. This requires an internet connection on the first run only.

```python
# This happens automatically during training:
from qai_hub_models.models.ddrnet23_slim import Model
model = Model.from_pretrained()  # Auto-downloads ~22MB weights
```

### Manual download (optional)

If you prefer to download weights manually (e.g., for offline environments):

| Source | Weights | Link |
|--------|---------|------|
| Official (Cityscapes) | DDRNet23-Slim (val mIoU 77.8%) | [Google Drive](https://drive.google.com/file/d/1d_K3Af5fKHYwxSo8HkxpnhiekhwovmiP/view) |
| Official (ImageNet backbone) | DDRNet23-Slim (top-1 err 29.8%) | [Google Drive](https://drive.google.com/file/d/1mg5tMX7TJ9ZVcAiGSB4PEihPtrJyalB4/view) |

### INT8 Safety

DDRNet23-Slim uses only Conv+BN+ReLU operations, making it fully compatible with INT8 quantization on the Qualcomm IQ-9075 Hexagon NPU. Unlike EfficientViT, there is no accuracy degradation after quantization.

---

## EfficientViT-B1 (Legacy)

> **Warning**: EfficientViT works well in FP32 training but suffers from INT8 quantization issues on the Qualcomm NPU. The Smooth Ground class collapses to 0% IoU after quantization. Use DDRNet23-Slim for IQ-9075 deployment.

| Item | Value |
|------|-------|
| File | `efficientvit_seg_b1_cityscapes.pt` |
| Model | EfficientViT-B1 Segmentation |
| Pretrained on | Cityscapes (19 classes) |
| Parameters | 4.8M |
| Cityscapes mIoU | 80.5% |
| Source | [mit-han-lab/efficientvit](https://github.com/mit-han-lab/efficientvit) |
| License | Apache 2.0 |

### Download

Pretrained segmentation models are listed at the [EfficientViT Segmentation README](https://github.com/mit-han-lab/efficientvit/blob/master/applications/efficientvit_seg/README.md#pretrained-efficientvit-segmentation-models).

Direct download link: [efficientvit_seg_b1_cityscapes.pt](https://huggingface.co/han-cai/efficientvit-seg/resolve/main/efficientvit_seg_b1_cityscapes.pt)

Place the downloaded file in this directory:

```
assets/
└── efficientvit_seg_b1_cityscapes.pt
```

### Usage

During training, the Cityscapes 19-class segmentation head is automatically replaced with a 7-class head matching the unified ontology. All backbone weights are loaded and the entire model is fine-tuned end-to-end.

The weight file path is resolved automatically when placed in this directory. No additional configuration is needed.