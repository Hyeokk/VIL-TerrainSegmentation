# Model Selection

## Overview

| | EfficientViT-B1 | DDRNet23-Slim |
|---|---|---|
| Architecture | Vision Transformer | CNN (Dual-Resolution) |
| Parameters | 4.8M | 5.7M |
| Core operations | Softmax, LayerNorm, GELU | Conv + BatchNorm + ReLU |
| Cityscapes mIoU (FP32) | 80.5% | 77.8% |
| INT8 quantization | Failed | Safe (< 1% loss) |
| NPU op coverage | Partial (unsupported ops) | 131/131 (100%) |
| IQ-9075 latency | Not deployable | ~15ms (60+ FPS) |

EfficientViT-B1 was initially selected for its higher FP32 accuracy. Training and ONNX export succeeded, but INT8 quantization for the IQ-9075 NPU caused critical accuracy loss. DDRNet23-Slim was adopted as the replacement because its CNN operations are inherently INT8-safe and fully verified on Qualcomm AI Hub.

The 2.7% mIoU gap on Cityscapes 19-class is expected to narrow on the simpler 7-class off-road task. More importantly, DDRNet23-Slim works after INT8 quantization; EfficientViT-B1 does not.

---

## EfficientViT-B1: Why It Failed

### Deployment Timeline

| Step | Result |
|------|--------|
| PyTorch training | Success (46.6 FPS on host GPU) |
| ONNX export (opset 17) | Success (18.5 MB, diff < 1e-4) |
| ONNX simplification | Success |
| Calibration (500 images) | Success |
| INT8 quantization | Completed |
| Accuracy verification | **Failed: Smooth Ground class collapse** |

### Root Cause

Three core ViT operations are structurally incompatible with INT8:

| Operation | Problem |
|-----------|---------|
| Softmax | Output follows a power-law distribution (most values near 0, few near 1). INT8 has only 256 levels and cannot represent this distribution. Distorted attention weights break the patch cooperation needed for classifying large uniform areas. |
| LayerNorm | Inter-channel activation ranges differ by 10x+. Layer-wise INT8 quantization applies a single scale factor to all channels, destroying low-range channels that may encode road texture features. |
| GELU | Non-linear attenuation in the negative domain cannot be approximated in INT8. Distorted MLP outputs cascade through subsequent layers. |

Smooth Ground is affected first because it has low-contrast, uniform texture similar to adjacent classes (Rough Ground, Vegetation), relying on fine-grained attention to distinguish. Classes with distinctive color signatures (Sky, Water, Vegetation) are relatively preserved.

### Qualcomm HTP Constraints

- Supports only INT8 or UINT16; no per-layer FP32 fallback
- FP32 Softmax/LayerNorm requires NPU-CPU data transfers, degrading real-time performance
- QNN SDK does not support ViT-specific quantization techniques (ShiftMax, ShiftGELU, I-LayerNorm)

### Attempted Mitigations

| Method | Accuracy Recovery | Speed Impact | Conclusion |
|--------|------------------|-------------|-----------|
| UINT16 Mixed Precision | Medium-High | -40~50% | 30 FPS target uncertain |
| QAT | Medium | None | ViT QAT is research-stage |
| CPU Fallback (Softmax/LN) | High | -60~70% | Defeats NPU deployment purpose |

**Conclusion**: Stable INT8 real-time inference on IQ-9075 with the ViT architecture is not feasible with current tooling.

---

## DDRNet23-Slim: Why It Works

### INT8-Native Operations

| Operation | INT8 Behavior |
|-----------|--------------|
| Conv + BN Fusion | BN folds into Conv as constants at inference -- disappears from quantization entirely |
| ReLU | `max(0, x)` is lossless in INT8, unlike Softmax (power-law) or GELU (negative non-linearity) |
| No Attention | No Softmax-based attention weights to distort |

### Dual-Resolution Architecture

DDRNet maintains two parallel branches with bidirectional fusion:

| Branch | Role | Examples |
|--------|------|----------|
| High-resolution | Spatial detail | Road boundaries, puddle edges, obstacle contours |
| Low-resolution | Semantic context | Wide-area classification (Smooth Ground, Sky) |

This design is particularly effective for distinguishing the subtle texture differences between Smooth Ground and Rough Ground.

---

## Implementation Note: qai_hub_models Wrapper Bypass

The `qai_hub_models` package wraps the raw DDRNet in a container whose `forward()` includes ImageNet normalization:

```
qai_hub_models.Model (wrapper)
    .model = raw DDRNet
    .forward(x) = self.model(normalize(x))
```

This causes two problems during training:

| Problem | Cause |
|---------|-------|
| Device mismatch error | `normalize()` creates mean/std tensors on CPU while input is on CUDA |
| Double normalization | `dataset.py` already applies ImageNet normalization |

**Fix**: `_extract_core_model()` in `models_ddrnet.py` extracts `model.model` (the raw DDRNet), bypassing the wrapper entirely.