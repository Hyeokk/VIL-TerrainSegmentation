# Pretrained Weights

This directory contains pretrained model weights used for fine-tuning. These files are not included in the repository and must be downloaded manually.

## EfficientViT-B1 (Cityscapes)

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