# Model Selection: EfficientViT-B1 to DDRNet23-Slim

## EfficientViT-B1 Failure

### Timeline

EfficientViT-B1 (4.8M params, Cityscapes 80.5% mIoU) was initially selected for its high FP32 accuracy and small parameter count. PyTorch training completed successfully and the model ran at 46.6 FPS on the host GPU.

| Step | Result |
|------|--------|
| PyTorch training | Success |
| ONNX export (opset 17) | Success (18.5 MB, diff < 1e-4) |
| ONNX simplification | Success |
| Calibration (500 images) | Success |
| INT8 quantization | Completed |
| Accuracy verification | **Failed: Smooth Ground collapse** |

### Root Cause: ViT Operations Are Incompatible with INT8

Three core operations in EfficientViT-B1 are structurally incompatible with INT8 quantization.

**Softmax**: Outputs follow a power-law distribution -- most values near 0, few near 1. INT8 has only 256 levels and cannot faithfully represent this non-uniform distribution. When attention weights are distorted, the patch cooperation required to classify large uniform areas like Smooth Ground breaks down.

**LayerNorm**: Inter-channel activation ranges differ by 10x or more. INT8 layer-wise quantization applies a single scale factor to all channels, destroying information in low-range channels. If those channels encode road texture features, the information is irrecoverable.

**GELU**: The non-linear attenuation in the negative domain cannot be precisely approximated in INT8. Distorted MLP outputs cascade through subsequent layers, causing compounding feature extraction failures.

### Why Smooth Ground Fails First

Smooth Ground has low-contrast, uniform texture and is visually similar to adjacent classes (Rough Ground, Vegetation). In FP32, the attention mechanism captures subtle texture differences with high precision. INT8 quantization degrades these fine-grained attention weights first. By contrast, Sky (blue), Water (reflections), and Vegetation (green) have distinctive color signatures and are relatively preserved.

### Qualcomm HTP Constraints

- HTP supports only INT8 or UINT16; no per-layer FP32 fallback
- Mixed-precision with FP32 Softmax/LayerNorm requires NPU-CPU data transfers, degrading real-time performance
- QNN SDK does not support ViT-specific quantization techniques (ShiftMax, ShiftGELU, I-LayerNorm) from research literature

### Attempted Mitigations

| Method | Accuracy Recovery | Speed Impact | Conclusion |
|--------|------------------|-------------|-----------|
| UINT16 Mixed Precision | Medium-High | -40~50% | 30 FPS target uncertain |
| QAT | Medium | None | ViT QAT is research-stage |
| CPU Fallback (Softmax/LN) | High | -60~70% | Defeats NPU deployment purpose |

Conclusion: achieving stable INT8 real-time inference on the IQ-9075 while retaining the ViT architecture is not feasible with current tooling.

---

## Why DDRNet23-Slim

### CNN Operations Are INT8-Native

| Operation | INT8 Behavior |
|-----------|--------------|
| Conv + BN Fusion | BatchNorm folds into Conv as constants at inference. BN disappears from quantization -- zero information loss. |
| ReLU | `max(0, x)` is lossless in INT8. Fundamentally different from Softmax (power-law) or GELU (negative non-linearity). |
| No Attention | No Softmax-based attention weights to distort. |

### Dual-Resolution Architecture

DDRNet maintains two parallel branches:

- **High-resolution branch**: preserves spatial detail (road boundaries, puddle edges, obstacle contours)
- **Low-resolution branch**: captures semantic context (wide-area classification: Smooth Ground, Sky)
- **Bidirectional fusion**: information exchange between branches

This design is particularly effective for distinguishing the subtle texture differences between Smooth Ground and Rough Ground.

### Qualcomm AI Hub Verification

DDRNet23-Slim is included in the official `qai_hub_models` package. All 131 operations execute on the NPU with zero CPU fallback. INT8 PTQ accuracy loss is under 1%. Benchmarked at 25.6ms on QCS8275; estimated ~15ms (60+ FPS) on IQ-9075 (100 TOPS).

### Model Comparison

| Item | EfficientViT-B1 | DDRNet23-Slim |
|------|----------------|---------------|
| Architecture | Vision Transformer | CNN (Dual-Resolution) |
| Parameters | 4.8M | 5.7M |
| Core ops | Softmax, LayerNorm, GELU | Conv + BatchNorm + ReLU |
| Cityscapes mIoU (FP32) | 80.5% | 77.8% |
| INT8 quantization | Failed | Safe (< 1% loss) |
| NPU op coverage | Partial | 131/131 (100%) |
| IQ-9075 latency | Not deployable | ~15ms (60+ FPS) |

The 2.7pp mIoU gap on Cityscapes 19-class is expected to narrow on the simpler 7-class off-road task. More importantly, DDRNet23-Slim works after INT8 quantization; EfficientViT-B1 does not.

### qai_hub_models Wrapper Bypass

The `qai_hub_models` package wraps the raw DDRNet in a container class whose `forward()` includes its own ImageNet normalization:

```
qai_hub_models.Model (wrapper)
    .model = raw DDRNet (Conv+BN+ReLU)
    .forward(x) = self.model(normalize(x))   <-- creates mean/std on CPU, conflicts with CUDA tensors
```

Problems:
1. GPU training: input is on CUDA but normalize creates mean/std on CPU -> device mismatch error
2. `dataset.py` already applies ImageNet normalization -> double normalization

Fix: `_extract_core_model()` in `models_ddrnet.py` extracts `model.model` (the raw DDRNet), bypassing the wrapper entirely.