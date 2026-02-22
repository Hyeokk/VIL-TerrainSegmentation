# Off-Road Semantic Segmentation for Tracked Robot

Real-time semantic segmentation for a caterpillar-track autonomous robot on unpaved terrain. Trained on three open-source off-road datasets under a unified 7-class ontology and deployed to a Qualcomm edge NPU for onboard traversability estimation.

```
Camera (1280x1080) --> DDRNet23-Slim (INT8) --> 7-class mask --> Traversability map
                       Qualcomm IQ-9075 NPU
                       ~15ms/frame (60+ FPS)
```

No target-camera data is used during training. Domain gap is addressed through photometric augmentation.

---

## Table of Contents

1. [Model Selection](#model-selection)
2. [7-Class Ontology](#7-class-ontology)
3. [Datasets](#datasets)
4. [Training](#training)
5. [Deployment (IQ-9075)](#deployment-iq-9075)
6. [Project Structure](#project-structure)
7. [Known Issues](#known-issues)
8. [License](#license)
9. [References](#references)

---

## Model Selection

| | EfficientViT-B1 | **DDRNet23-Slim** |
|---|---|---|
| Architecture | Vision Transformer | CNN (Dual-Resolution) |
| Parameters | 4.8M | 5.7M |
| Cityscapes mIoU (FP32) | 80.5% | 77.8% |
| INT8 quantization | Smooth Ground collapse (0%) | No degradation |
| Qualcomm NPU coverage | Softmax/LN/GELU unsupported | 131/131 ops on NPU |
| AI Hub verified | No | Yes |

EfficientViT-B1 was originally selected for higher FP32 accuracy. However, its core operations (Softmax, LayerNorm, GELU) produce distributions that INT8 cannot faithfully represent. On the Qualcomm Hexagon NPU, this causes Smooth Ground to collapse to 0% IoU -- the most critical class for path planning.

DDRNet23-Slim uses only Conv + BatchNorm + ReLU. BatchNorm folds into Conv at inference (zero quantization overhead), and `max(0, x)` is lossless in INT8. All 131 operations execute on the NPU with no CPU fallback.

---

## 7-Class Ontology

| ID | Class | Description | Traversability |
|----|-------|-------------|----------------|
| 0 | Smooth Ground | Asphalt, concrete, packed dirt | Optimal path |
| 1 | Rough Ground | Sand, gravel, mud, snow | Passable (reduce speed) |
| 2 | Vegetation | Low grass, moss, leaves | Passable (tracks) |
| 3 | Obstacle | Trees, rocks, buildings, fences, bushes | Avoid |
| 4 | Water | Puddles, streams, lakes | Avoid (flood risk) |
| 5 | Sky | Sky, clouds | Ignore |
| 6 | Dynamic | People, vehicles, animals | Avoid (safety) |

Caterpillar-specific mappings that differ from standard autonomous driving:

- `bush` --> Obstacle (track entanglement risk)
- `puddle` --> Water (unknown depth, drivetrain flooding risk)
- `dirt` --> Smooth Ground (optimal surface for tracks)
- `grass` --> Vegetation (low grass is traversable by tracks)

Full per-dataset class mappings are defined in `src/dataset.py`.

---

## Datasets

| Dataset | Train Images | Environment | Original Classes | License |
|---------|-------------|-------------|-----------------|---------|
| RELLIS-3D | 4,169 | Military test trails (US) | 20 --> 7 | CC BY-NC-SA 3.0 |
| RUGD | 7,436 | Parks, trails, forests | 24 --> 7 | CC BY 4.0 |
| GOOSE | 7,845 | European outdoor/forest | 64 --> 7 | CC BY-SA 4.0 |
| **Total** | **19,450** | | | |

Validation set: 1,788 images (RELLIS-3D 30% test split).

All datasets are merged via `ConcatDataset` after remapping each original ontology to the 7-class scheme. RELLIS-3D uses integer ID lookup, RUGD uses RGB color mapping, GOOSE uses CSV name mapping. Each dataset must be downloaded separately under `data/` (see `data/README.md`). RELLIS-3D is required; RUGD and GOOSE are optional and auto-detected at runtime.

---

## Training

### Quick Start

```bash
# 1. Environment
bash setup.sh && conda activate offroad

# 2. Data preparation
# Download datasets into data/ (see data/README.md)
python scripts/make_split_custom.py          # 70/30 train/test split
python scripts/preprocess_datasets.py        # fast mode (~2 min, one-time)

# 3. Verify
python scripts/verify_all.py

# 4. Train
python scripts/train.py --model ddrnet23-slim --fast --num_workers 8

# 5. Export for deployment
python scripts/export_qnn.py \
    --checkpoint ./checkpoints/ddrnet23-slim/best_model.pth \
    --method hub
```

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model | DDRNet23-Slim (5.7M) | INT8-safe, Qualcomm AI Hub verified |
| Pretrained | Cityscapes 19-class | Transfer low-level visual features |
| Input size | 544 x 640 (H x W) | 0.5x of S10 Ultra (1080 x 1280), same aspect ratio |
| Batch size | 8 | 24GB VRAM constraint; ~2.8M pixels/batch |
| Epochs | 200 | |
| Optimizer | AdamW (weight_decay=0.01) | Decoupled weight decay |
| Backbone LR | 1e-4 | Preserve pretrained features (10x lower than head) |
| Head LR | 1e-3 | Fast convergence for new 7-class head |
| LR schedule | 5-ep linear warmup + 195-ep cosine annealing | Stabilize Xavier-init head, then smooth decay to zero |
| Loss | Focal Loss (gamma=2.0) | `(1-p_t)^2` auto-suppresses easy/large-area classes |
| EMA | decay=0.9999 | Half-life ~2.9 epochs; smooths batch noise |
| AMP | FP16 mixed precision | ~1.5x throughput on Tensor Cores; FP32 master weights |
| Gradient clipping | max_norm=5.0 | Safety net for AMP + class-imbalanced batches |

### Differential Learning Rate

The pretrained backbone and the Xavier-initialized head require different learning rates. A uniform LR=1e-3 destroys backbone features within 2 epochs -- the loss drops 89% in epoch 1-to-2, which indicates feature collapse rather than learning. The head then has nothing to classify, defaulting to the majority class (mode collapse).

```
Backbone (161 tensors):  lr x 0.1 = 1e-4   --> slow adaptation, preserves edges/textures
Head     (7 tensors):    lr       = 1e-3   --> fast learning for 7-class mapping
```

The 10x ratio is maintained across warmup and cosine annealing. Configurable via `--bb_lr_factor`:

```bash
python scripts/train.py --bb_lr_factor 0.1 ...    # default (10x ratio)
python scripts/train.py --bb_lr_factor 0.05 ...   # 20x ratio
```

### Focal Loss

```
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
```

Gamma and alpha address separate problems:

| Mechanism | Parameter | Effect |
|-----------|-----------|--------|
| Difficulty balancing | gamma=2.0 | Well-classified pixels (high p_t) contribute near-zero loss |
| Safety weighting | alpha per class | Rare safety-critical classes receive stronger gradient |

Per-class alpha values:

| Class | Alpha | Rationale |
|-------|-------|-----------|
| Smooth Ground | 1.5 | Core navigation path |
| Rough Ground | 3.0 | Rare, affects speed decisions |
| Vegetation | 1.0 | Baseline (gamma handles area suppression) |
| Obstacle | 1.0 | Baseline (abundant, diverse) |
| Water | 5.0 | Very rare, safety-critical |
| Sky | 1.0 | Baseline (gamma handles area suppression) |
| Dynamic | 5.0 | Very rare, safety-critical |

Design constraint: minimum alpha is 1.0. Values below 1.0 (e.g., Sky=0.3) starve gradient during early training and cause mode collapse.

**Implementation constraint:** Alpha must be applied **outside** `F.cross_entropy`. Passing it as `weight=` corrupts the focal term: `pt = exp(-alpha * CE) = p_t^alpha` instead of `p_t`. See `src/dataset.py` for the correct implementation.

### Data Augmentation

Training: `RandomScale(0.5-2.0) --> Pad --> RandomCrop(544x640) --> HFlip --> PhotometricAug --> Normalize`

Validation: `Resize(544x640) --> Normalize`

| Augmentation | Parameters | Purpose |
|-------------|-----------|---------|
| ColorJitter | brightness/contrast/saturation=0.4, hue=0.15 | Camera domain gap |
| Gaussian Blur | p=0.3, radius=0.5-2.0 | Motion blur simulation |
| Random Shadow | p=0.2, darkening=0.3-0.7 | Tree canopy shadow simulation |
| Random Erasing | p=0.3, scale=0.02-0.2 | Occlusion robustness |
| Random Grayscale | p=0.05 | Low-light conditions |

Geometric distortion is not applied -- handled by camera calibration at deployment time.

### Fast Mode

Pre-resizes original images (up to 1920px PNG, ~5MB) to max-1024px JPEG (quality=95, ~100KB). Labels use NEAREST interpolation to preserve class IDs. Reduces epoch time from ~470s to ~180s with no impact on training quality.

```bash
python scripts/preprocess_datasets.py    # one-time, ~2 min
python scripts/train.py --fast ...
```

### CLI Reference

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | ddrnet23-slim | Model architecture |
| `--batch_size` | 8 | Batch size |
| `--epochs` | 200 | Total epochs |
| `--lr` | 0.001 | Head learning rate |
| `--bb_lr_factor` | 0.1 | Backbone LR = lr x bb_lr_factor |
| `--warmup_epochs` | 5 | Linear warmup duration |
| `--crop_size` | 544,640 | Training crop size (H,W) |
| `--eval_interval` | 5 | Validation frequency (epochs) |
| `--fast` | off | Use pre-resized datasets |
| `--num_workers` | 4 | DataLoader workers |
| `--quiet` | off | One summary line per epoch |
| `--resume` | None | Resume from checkpoint path |
| `--no_amp` | off | Disable mixed precision |

---

## Deployment (IQ-9075)

### Pipeline Overview

```
STAGE 1  Training (host PC, RTX PRO 4000)
         python scripts/train.py --model ddrnet23-slim --fast --num_workers 8
         --> checkpoints/ddrnet23-slim/best_model.pth

STAGE 2  ONNX export (host PC)
         python scripts/export_qnn.py --checkpoint best_model.pth --method onnx
         --> deploy/ddrnet23_slim_unified7class_544x640.onnx

STAGE 3  INT8 quantization + compile (AI Hub cloud or QNN SDK)
         python scripts/export_qnn.py --checkpoint best_model.pth --method hub
         --> deploy/ddrnet23_slim_int8.bin

STAGE 4  Device inference (IQ-9075)
         python scripts/infer_qnn_video.py --model ddrnet23_slim_int8.bin --input video.mp4
         python scripts/infer_qnn_ros2.py  --model ddrnet23_slim_int8.bin --topic /camera/image_raw
```

### Export Methods

| Method | Flag | Requirements | Notes |
|--------|------|-------------|-------|
| AI Hub (cloud) | `--method hub` | `qai-hub` package + API token | Recommended starting point |
| ONNX only | `--method onnx` | None | Host-PC testing |
| Local QNN SDK | `--method local` | QNN SDK installed | Advanced |

### Inference on Device

Video file:

```bash
# Host PC test (ONNX + CPU)
python scripts/infer_qnn_video.py \
    --model deploy/ddrnet23_slim_unified7class_544x640.onnx \
    --input video.mp4 --output result.mp4 --backend cpu

# IQ-9075 (QNN binary)
python scripts/infer_qnn_video.py \
    --model deploy/ddrnet23_slim_int8.bin \
    --input video.mp4 --output result.mp4
```

ROS2 live camera:

```bash
python3 scripts/infer_qnn_ros2.py \
    --model deploy/ddrnet23_slim_int8.bin \
    --topic /camera/s10_ultra/color/image_raw
```

Published ROS2 topics:

| Topic | Type | Content |
|-------|------|---------|
| `~/segmentation` | sensor_msgs/Image (mono8) | 7-class mask (pixel values 0-6) |
| `~/costmap` | nav_msgs/OccupancyGrid | Navigation costmap (0=free, 100=lethal) |
| `~/overlay` | sensor_msgs/Image (bgr8) | Original + segmentation blend |

### Preprocessing Requirement

The exported model does not include normalization internally. All inference paths must apply ImageNet normalization before the model:

```python
tensor = frame.astype(np.float32) / 255.0
tensor = (tensor - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
```

This is standard for Qualcomm NPU deployment -- normalization (division, subtraction) is more efficient on CPU, while the NPU is optimized for Conv/BN/ReLU. All provided inference scripts include this step.

---

## Training Environment

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX PRO 4000 Blackwell (24GB GDDR7, sm_120) |
| CPU | AMD Ryzen 9700X |
| RAM | 64GB DDR5 |
| Storage | Samsung 990 PRO NVMe |
| Framework | PyTorch 2.10.0, CUDA 12.8 |
| Camera | MRDVS S10 Ultra RGB-D (1280 x 1080) |
| Deploy target | Qualcomm Dragonwing IQ-9075 (100 TOPS NPU, INT8) |

---

## Project Structure

```
VIL-TerrainSegmentation/
├── scripts/
│   ├── train.py                   # Training (differential LR, Focal Loss)
│   ├── evaluate.py                # mIoU evaluation
│   ├── infer_cam.py               # PyTorch image/video inference
│   ├── infer_qnn_video.py         # IQ-9075 video inference (ONNX/QNN)
│   ├── infer_qnn_ros2.py          # IQ-9075 ROS2 live inference
│   ├── export_onnx.py             # Generic ONNX export
│   ├── export_qnn.py              # QNN export for IQ-9075
│   ├── preprocess_datasets.py     # Fast mode preprocessing
│   ├── make_split_custom.py       # 70/30 split generation
│   ├── visualize_predictions.py   # Prediction visualization
│   └── verify_all.py              # Environment verification
├── src/
│   ├── dataset.py                 # Datasets, class mappings, Focal Loss, EMA
│   ├── models.py                  # Model factory (DDRNet / EfficientViT / FFNet)
│   └── models_ddrnet.py           # DDRNet23-Slim builder (qai_hub_models)
├── data/                          # Datasets (download separately)
│   ├── Rellis-3D/
│   ├── RUGD/
│   └── GOOSE/
├── checkpoints/                   # Training checkpoints
├── deploy/                        # ONNX and QNN export artifacts
├── configs/
│   └── rellis3d_unified.yaml
├── docs/
│   └── progress_report_kr.md      # Detailed progress report (Korean)
└── setup.sh
```

DDRNet23-Slim pretrained weights are auto-downloaded by `qai_hub_models`. For legacy EfficientViT weights, see `assets/README.md`.

---

## Known Issues

1. **Focal Loss alpha must not use `weight=` in `F.cross_entropy`.** This corrupts focal modulation (`p_t` becomes `p_t^alpha`). Alpha is applied as a separate post-CE multiplication. See `src/dataset.py`.

2. **DDRNet output resolution may differ from input.** Some variants output at 1/8 scale. The wrapper and inference scripts handle upsampling via `F.interpolate`, but custom code must verify output shape.

3. **IQ-9075 may not appear in AI Hub device catalog.** Use the proxy device name: `hub.Device("Qualcomm QCS9075 (Proxy)")`.

4. **EfficientViT code is retained as legacy.** `verify_all.py` treats DDRNet as PRIMARY (error if missing) and EfficientViT/FFNet as LEGACY (warning only).

---

## License

Training code (`scripts/`, `src/`, `setup.sh`): **MIT License**.

| Component | License | Source |
|-----------|---------|--------|
| DDRNet | MIT | [ydhongHIT/DDRNet](https://github.com/ydhongHIT/DDRNet) |
| qai_hub_models | BSD-3 | [qualcomm/ai-hub-models](https://github.com/qualcomm/ai-hub-models) |
| EfficientViT | Apache 2.0 | [mit-han-lab/efficientvit](https://github.com/mit-han-lab/efficientvit) |
| RELLIS-3D | CC BY-NC-SA 3.0 | [unmannedlab/RELLIS-3D](https://github.com/unmannedlab/RELLIS-3D) |
| RUGD | CC BY 4.0 | [rugd.vision](http://rugd.vision/) |
| GOOSE | CC BY-SA 4.0 | [goose-dataset.de](https://goose-dataset.de/) |

RELLIS-3D uses CC BY-NC-SA 3.0 (non-commercial). Weights trained with RELLIS-3D inherit this restriction. For commercial deployment, train with RUGD + GOOSE only.

---

## References

```bibtex
@article{hong2021ddrnet,
  title={Deep Dual-resolution Networks for Real-time and Accurate Semantic Segmentation of Road Scenes},
  author={Hong, Yuanduo and Pan, Huihui and Sun, Weichao and Jia, Yisong},
  journal={arXiv preprint arXiv:2101.06085},
  year={2021}
}

@inproceedings{lin2017focal,
  title={Focal Loss for Dense Object Detection},
  author={Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
  booktitle={ICCV},
  year={2017}
}

@inproceedings{cai2023efficientvit,
  title={EfficientViT: Lightweight Multi-Scale Attention for High-Resolution Dense Prediction},
  author={Cai, Han and Li, Junyan and Hu, Muyan and Gan, Chuang and Han, Song},
  booktitle={ICCV},
  year={2023}
}

@inproceedings{jiang2021rellis3d,
  title={RELLIS-3D Dataset: Data, Benchmarks and Analysis},
  author={Jiang, Peng and Osteen, Philip and Wigness, Maggie and Saripalli, Srikanth},
  booktitle={ICRA},
  year={2021}
}

@inproceedings{wigness2019rugd,
  title={A RUGD Dataset for Autonomous Navigation and Visual Perception in Unstructured Outdoor Environments},
  author={Wigness, Maggie and Eum, Sungmin and Rogers, John G and Han, David and Kwon, Heesung},
  booktitle={IROS},
  year={2019}
}

@inproceedings{mortimer2024goose,
  title={The GOOSE Dataset for Perception in Unstructured Environments},
  author={Mortimer, Peter and others},
  booktitle={ICRA},
  year={2024}
}
```