# DDRNet23-Slim Training-to-Deployment Pipeline Review

## Final Review for Qualcomm Dragonwing IQ-9075 Target


## 1. Background: EfficientViT to DDRNet Transition

| Item | EfficientViT-B1 | DDRNet23-Slim |
|------|-----------------|---------------|
| Cityscapes mIoU | 80.5% | 77.8% |
| Parameters | 4.8M | 5.7M |
| Core operations | **Attention + LayerNorm** | **Conv + BN + ReLU only** |
| INT8 quantization | Smooth Ground class collapses | Conv/BN/ReLU stable under INT8 |
| NPU compatibility | ViT ops fall back to CPU | All ops run on NPU |
| Qualcomm AI Hub | Not verified | Officially verified |

**Key reason for transition**: EfficientViT-B1's Attention operations cause activation distribution distortion under INT8 quantization, leading to specific class collapse (Smooth Ground drops to 0% IoU). DDRNet23-Slim uses only quantization-friendly operations.


## 2. Full Pipeline Overview

```
+---------------------------------------------------------------------+
|  STAGE 1: Training (RTX PRO 4000 Blackwell, CUDA 12.8)             |
|  qai_hub_models DDRNet23-Slim (Cityscapes 19-class pretrained)      |
|       -> Head replacement: 19 -> 7 classes                          |
|  3 datasets (19,450 images) + Focal Loss + AdamW + Cosine Scheduler |
|  200 epochs, FP16 AMP -> best_model.pth (EMA weights)              |
+---------------------------------------------------------------------+
|  STAGE 2: ONNX Export                                               |
|  best_model.pth -> torch.onnx.export (opset 17)                    |
|       -> Input: (1, 3, 544, 640), Output: (1, 7, 544, 640)        |
|       -> ddrnet23_slim_7class_544x640.onnx                         |
+---------------------------------------------------------------------+
|  STAGE 3: QNN Quantization + Compile (AI Hub or QNN SDK)           |
|  ONNX -> QNN Context Binary (INT8 W8A8)                            |
|       -> Calibration: 200-500 image subset from training data       |
|       -> ddrnet23_slim_int8.bin                                     |
+---------------------------------------------------------------------+
|  STAGE 4: IQ-9075 NPU Deployment                                   |
|  S10 Ultra (1280x1080) -> Resize (640x544) -> NPU inference        |
|       -> ~25-35ms/frame (~30 FPS)                                   |
|  7-class segmentation map -> Costmap -> Navigation decision         |
+---------------------------------------------------------------------+
```


## 3. Detailed Review by Stage

### 3.1 Model: DDRNet23-Slim via qai_hub_models

**Selection rationale:**
- Compile/profile verified on IQ-9075 (QCS9075) through Qualcomm AI Hub
- `Model.from_pretrained()` auto-loads Cityscapes pretrained weights
- Internal architecture optimized for Qualcomm NPU (align_corners=False, nearest upsample, etc.)
- Built-in ONNX/QNN export scripts

**Head replacement strategy (same as EfficientViT):**
```
Original: DDRNet23-Slim (Cityscapes 19-class)
  -- final_layer: Conv2d(64, 19, 1)    <- Cityscapes 19 classes
                         | replace
Modified: DDRNet23-Slim (Off-road 7-class)
  -- final_layer: Conv2d(64, 7, 1)     <- Unified 7 classes
  -- Xavier initialization (head only, backbone preserved)
```

**INT8 safety verification:**

| Operation | Used in DDRNet23-Slim | INT8 Stability |
|-----------|----------------------|----------------|
| Conv2d | Yes (all layers) | Stable |
| BatchNorm2d | Yes (all layers) | Foldable into Conv |
| ReLU | Yes (all layers) | Fully stable |
| Bilinear Upsample | Yes (decoder) | Stable |
| Element-wise Add | Yes (skip connections) | Stable |
| Attention | Not present | N/A |
| LayerNorm | Not present | N/A |
| GELU/SiLU | Not present | N/A |

All operations are INT8-stable. BN folding further reduces inference compute.

### 3.2 Training Data

| Item | Setting | Review |
|------|---------|--------|
| Training data | 19,450 images (RELLIS + RUGD + GOOSE) | Covers diverse environments |
| Validation data | 1,788 images (RELLIS-3D test 30%) | Adequate |
| Class mapping | 7-class unified ontology | Reflects caterpillar traversability |
| ignore_index | 255 | Excludes void/undefined |

### 3.3 Training Pipeline

| Item | Setting | Review |
|------|---------|--------|
| Input resolution | 544x640 (HxW) | Half of S10 Ultra, multiple of 32 |
| RandomScale | 0.5~2.0 | Multi-scale training |
| RandomCrop | 544x640 | Fixed batch size |
| ColorJitter | b=0.4, c=0.4, s=0.4, h=0.15 | Camera domain gap correction |
| Random Erasing | p=0.3 | Occlusion robustness |
| Loss | Focal Loss (gamma=2.0) | Handles class imbalance |
| Class weights | Water=5.0, Dynamic=5.0 | Emphasizes safety-critical classes |
| Optimizer | AdamW (lr=1e-3, wd=0.01) | OK |
| Scheduler | Warmup 20ep + Cosine 180ep | OK |
| EMA | decay=0.9999 | Training stabilization |
| AMP | FP16 | Leverages Blackwell Tensor Cores |
| Gradient Clipping | max_norm=5.0 | OK |
| Batch Size | 8 | Fits 24GB GDDR7 |
| Epochs | 200 | OK |

### 3.4 Validation Pipeline

| Item | Setting | Review |
|------|---------|--------|
| Input | Resize 544x640 (full image) | Resize, not crop |
| Metric | mIoU (7-class) | OK |
| Eval frequency | Every 5 epochs | OK |
| Eval model | EMA weights | OK |

### 3.5 ONNX Export

| Item | Setting | Review |
|------|---------|--------|
| Opset | 17 | QNN compatible (>=11 required, 17 recommended) |
| Input size | (1, 3, 544, 640) fixed | Fixed recommended for NPU optimization |
| dynamic_axes | Batch only | Fixed H/W is better for NPU |
| constant_folding | True | Enables BN folding optimization |

**Notes:**
- Must maintain `align_corners=False` (DDRNet official setting)
- `F.interpolate` with `mode='bilinear'` is supported by QNN
- DDRNet uses both bilinear and nearest interpolation; both supported by QNN

### 3.6 QNN Quantization (INT8 W8A8)

| Item | Setting | Review |
|------|---------|--------|
| Method | Post-Training Quantization (PTQ) | Sufficient (QAT not needed) |
| Precision | W8A8 (weight 8-bit, activation 8-bit) | Optimal for IQ-9075 NPU |
| Calibration | 200-500 training images | OK |
| Target runtime | qnn_context_binary | IQ-9075 Hexagon NPU |

**INT8 quantization risk checklist:**
- [x] No Attention operations -- safe
- [x] No LayerNorm -- safe
- [x] No GELU/SiLU -- safe
- [x] Sigmoid/Softmax only at final output -- argmax, no impact
- [x] BN folds into Conv -- no additional error
- [x] Skip connection Add -- small range difference, safe

### 3.7 IQ-9075 Deployment

| Item | Specification | Review |
|------|---------------|--------|
| NPU | Hexagon, 100 TOPS (INT8 Dense) | Sufficient for DDRNet |
| Memory | 36GB LPDDR5 | ~22MB model fits easily |
| Camera input | MIPI CSI-2 4-lane | S10 Ultra compatible |
| OS | Ubuntu/Yocto Linux | OK |
| AI framework | QNN, ONNX Runtime, TFLite | OK |
| Target latency | < 40ms/frame (>25 FPS) | ~25ms based on similar chipset benchmarks |

**On-device inference pipeline:**
```
S10 Ultra Camera (1280x1080 RGB)
    -> CPU: Resize to 640x544, normalize
    -> NPU: DDRNet23-Slim INT8 inference (~25ms)
    -> CPU: argmax -> 7-class segmentation map
    -> CPU: Costmap conversion (FREE/CAUTION/BLOCKED/HAZARD)
    -> MCU (Cortex-R52): Navigation control
```


## 4. Potential Risks and Mitigations

### Risk 1: qai_hub_models may only provide ImageNet backbone weights
- **Mitigation**: Check `out_channels` after `from_pretrained()`
  - 19 = Cityscapes segmentation weights (optimal)
  - 1000 = ImageNet classifier only (additional training needed)
- **Fallback**: Manually download official DDRNet Cityscapes weights

### Risk 2: Resolution 544x640 may be suboptimal for NPU
- **Mitigation**: IQ-9075 Hexagon NPU supports arbitrary resolutions
- **Alternative**: Can switch to 512x640 or 544x672 (powers of 2 multiples)
- **Current setting**: 544 (=32x17) x 640 (=32x20) -- suitable for stride-32 models

### Risk 3: Off-road domain gap
- **Mitigation**: 3 mixed datasets + strong color augmentation
- **Additional option**: Fine-tune with target camera (S10 Ultra) data after collection

### Risk 4: Unsupported operations during QNN conversion
- **Mitigation**: DDRNet is verified in qai_hub_models -- all operations confirmed supported
- **Caution**: Do not add custom operations (maintain standard Conv/BN/ReLU/Upsample only)


## 5. File Structure Changes

```
src/
    -- models.py              # Existing (EfficientViT + FFNet + DDRNet routing)
    -- models_ddrnet.py       # New (qai_hub_models-based DDRNet23-Slim)
    -- dataset.py             # No changes
scripts/
    -- train.py               # --model ddrnet23-slim support added
    -- export_onnx.py         # DDRNet compatible
    -- export_qnn.py          # New (IQ-9075 QNN export)
assets/
    -- README.md              # DDRNet weights info added
setup.sh                      # pip install qai-hub-models added
```


## 6. Execution Order

```bash
# 1. Environment setup (existing + qai-hub-models added)
bash setup.sh
conda activate offroad

# 2. Data preprocessing (same as before)
python scripts/preprocess_datasets.py

# 3. Train DDRNet23-Slim
python scripts/train.py --model ddrnet23-slim --fast --num_workers 8

# 4. ONNX export
python scripts/export_qnn.py \
    --checkpoint ./checkpoints/ddrnet23-slim/best_model.pth \
    --method onnx

# 5. QNN quantization + compile (via Qualcomm AI Hub or QNN SDK)
python scripts/export_qnn.py \
    --checkpoint ./checkpoints/ddrnet23-slim/best_model.pth \
    --method hub

# 6. Deploy to IQ-9075
# Transfer QNN Context Binary to IQ-9075 and run inference via QAIRT
```


## 7. Conclusion

**The DDRNet23-Slim + qai_hub_models pipeline is suitable for IQ-9075 deployment.**

| Check Item | Result |
|------------|--------|
| Is the model INT8 quantization safe? | Yes -- Conv+BN+ReLU only |
| Do all operations run on NPU? | Yes -- Qualcomm AI Hub verified |
| Are pretrained weights secured? | Yes -- qai_hub_models auto-download |
| Does training resolution match deployment? | Yes -- 544x640 (train = deploy) |
| Is class imbalance handled? | Yes -- Focal Loss + class weights |
| Is domain gap addressed? | Yes -- 3 datasets + color augmentation |
| Is real-time inference feasible? | Yes -- ~25-35ms estimated (>25 FPS) |