# Off-Road Semantic Segmentation for Tracked Robot

Training code for off-road semantic segmentation on a caterpillar-track autonomous robot. Combines three open-source datasets (RELLIS-3D, RUGD, GOOSE) into a unified 7-class ontology and fine-tunes a pretrained segmentation model for real-time traversability estimation.

- **Primary model**: [DDRNet23-Slim](https://github.com/ydhongHIT/DDRNet) (5.7M params) — INT8-safe, Qualcomm NPU verified
- **Legacy model**: [EfficientViT-B1](https://github.com/mit-han-lab/efficientvit) (4.8M params) — higher mIoU but INT8 quantization issues on NPU
- **No target camera data** is used during training — only open-source datasets.
- Inference target: MRDVS S10 Ultra RGB-D camera (1280x1080).
- Deployment target: Qualcomm Dragonwing IQ-9075 edge device (100 TOPS Hexagon NPU).

---

## Model Selection

| Model | Params | Cityscapes mIoU | Ops | INT8 Safe | NPU Verified |
|-------|--------|-----------------|-----|-----------|-------------|
| **DDRNet23-Slim** ★ | 5.7M | 77.8% | Conv+BN+ReLU | ✅ | ✅ Qualcomm AI Hub |
| EfficientViT-B1 | 4.8M | 80.5% | Attention+LN | ❌ Class collapse | ❌ |

**Why DDRNet23-Slim?** EfficientViT-B1 achieves higher accuracy in FP32, but its Attention and LayerNorm operations cause catastrophic accuracy loss (Smooth Ground class collapses to 0%) after INT8 quantization on the Qualcomm Hexagon NPU. DDRNet23-Slim uses only Conv+BN+ReLU — all INT8-safe operations with no quantization degradation.

See [docs/PIPELINE_REVIEW.md](docs/PIPELINE_REVIEW.md) for the full training-to-deployment pipeline review.

---

## 7-Class Ontology

| ID | Class | Traversability | ID | Class | Traversability |
|----|-------|----------------|----|-------|----------------|
| 0 | Smooth Ground | Optimal | 4 | Water | Avoid |
| 1 | Rough Ground | Slow down | 5 | Sky | Ignore |
| 2 | Vegetation | Passable | 6 | Dynamic | Avoid |
| 3 | Obstacle | Avoid | | | |

Notable caterpillar-specific mappings: bush → Obstacle (track entanglement risk), puddle → Water (flooding risk), dirt → Smooth Ground (optimal surface for tracks). Full mappings in `src/dataset.py`.

---

## Datasets

Each dataset must be downloaded separately under `data/`. RELLIS-3D is required; RUGD and GOOSE are optional and auto-detected.

| Dataset | Images | License | Link |
|---------|--------|---------|------|
| **RELLIS-3D** | 4,169 (train) | [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/) | [GitHub](https://github.com/unmannedlab/RELLIS-3D) |
| **RUGD** | 7,436 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | [rugd.vision](http://rugd.vision/) |
| **GOOSE** | 7,845 | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | [goose-dataset.de](https://goose-dataset.de/) |

All datasets are merged via `ConcatDataset` after mapping each original ontology to the unified 7-class scheme (RELLIS: ID lookup, RUGD: RGB color map, GOOSE: CSV name mapping).

---

## Training Strategy

**Resolution**: Training crop 544x640 (H,W) matches the S10 Ultra aspect ratio at ~0.5x scale.

**Pipeline**: `RandomScale(0.5~2.0) → Pad → RandomCrop(544x640) → HFlip → PhotometricAug → Normalize`

**Augmentation**: ColorJitter, GaussianBlur, RandomShadow, RandomGrayscale, RandomErasing — designed to bridge the domain gap between training cameras and the target S10 Ultra.

**Loss**: Focal Loss (gamma=2.0) with per-class weights emphasizing rare safety-critical classes (Water: 5.0, Dynamic: 5.0) and de-weighting common classes (Vegetation: 0.5, Sky: 0.3).

**Optimizer**: AdamW (lr=1e-3, wd=0.01) with 20-epoch LinearLR warmup followed by 180-epoch CosineAnnealing. EMA (decay=0.9999), FP16 AMP, gradient clipping (max_norm=5.0).

**Fast mode**: Pre-resizes images to max 1024px JPEG (~5MB PNG → ~100KB JPEG), reducing epoch time by ~2.5x with no quality loss.

---

## Training Environment

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX PRO 4000 Blackwell (24GB GDDR7) |
| Framework | PyTorch 2.10.0, CUDA 12.8 |

---

## Getting Started

```bash
# 1. Setup environment (includes DDRNet dependencies)
bash setup.sh && conda activate offroad

# 2. Place datasets under data/ (see data/README.md for download links)

# 3. Create train/test split (70/30)
python scripts/make_split_custom.py

# 4. Verify setup
python scripts/verify_all.py

# 5. (Recommended) Preprocess for fast training
python scripts/preprocess_datasets.py

# 6. Train DDRNet23-Slim (recommended for IQ-9075 deployment)
python scripts/train.py --model ddrnet23-slim --fast --num_workers 8

# 7. Evaluate
python scripts/evaluate.py --checkpoint ./checkpoints/ddrnet23-slim/best_model.pth

# 8. Export for IQ-9075 deployment
python scripts/export_qnn.py \
    --checkpoint ./checkpoints/ddrnet23-slim/best_model.pth \
    --method hub
```

### Legacy: EfficientViT-B1

EfficientViT models are still supported but not recommended for IQ-9075 INT8 deployment:

```bash
# Download pretrained weights first (see assets/README.md)
python scripts/train.py --model efficientvit-b1 --fast --num_workers 8
```

Adjust `--num_workers` and `--batch_size` according to your hardware. Use `--fast` flag if you ran step 5.

---

## Deployment Pipeline (IQ-9075)

```
DDRNet23-Slim (PyTorch, FP32)
  → ONNX export (opset 17)
  → QNN compile + INT8 quantize (Qualcomm AI Hub)
  → QNN Context Binary (IQ-9075 Hexagon NPU)
  → ~25-35ms/frame (~30 FPS)
```

Three export methods are available:

| Method | Command | Requires |
|--------|---------|----------|
| **AI Hub (cloud)** | `--method hub` | `qai-hub` + API token |
| **ONNX only** | `--method onnx` | Nothing extra |
| **Local QNN SDK** | `--method local` | QNN SDK installed |

See `python scripts/export_qnn.py --help` for details.

---

## Project Structure

```
VIL-TerrainSegmentation/
├── assets/                        # Pretrained weights (see assets/README.md)
├── configs/                       # Training config YAML
├── data/                          # Datasets (download separately, see data/README.md)
├── docs/
│   └── PIPELINE_REVIEW.md         # Full pipeline review (train → deploy)
├── scripts/
│   ├── train.py                   # Training (DDRNet + EfficientViT + FFNet)
│   ├── evaluate.py                # Evaluation (mIoU)
│   ├── infer_cam.py               # Image/video inference
│   ├── export_onnx.py             # ONNX export (generic)
│   ├── export_qnn.py              # QNN export for IQ-9075 deployment
│   ├── preprocess_datasets.py     # Fast mode preprocessing
│   ├── make_split_custom.py       # Split creation
│   ├── visualize_predictions.py   # Prediction visualization
│   └── verify_all.py              # Environment verification
├── src/
│   ├── dataset.py                 # Dataset loaders & class mappings
│   ├── models.py                  # Model factory (EfficientViT, FFNet)
│   └── models_ddrnet.py           # DDRNet23-Slim (qai_hub_models based)
└── setup.sh                       # Environment setup
```

`assets/` is not included in this repository. DDRNet23-Slim pretrained weights are auto-downloaded by `qai_hub_models`. For EfficientViT, see [assets/README.md](assets/README.md).

---

## License

Training code (`scripts/`, `src/`, `setup.sh`) is released under the **MIT License**.

| Component | License | Link |
|-----------|---------|------|
| DDRNet (model & weights) | MIT | [ydhongHIT/DDRNet](https://github.com/ydhongHIT/DDRNet) |
| qai_hub_models | BSD-3 | [qualcomm/ai-hub-models](https://github.com/qualcomm/ai-hub-models) |
| EfficientViT (model & weights) | Apache 2.0 | [mit-han-lab/efficientvit](https://github.com/mit-han-lab/efficientvit) |
| RELLIS-3D (dataset) | CC BY-NC-SA 3.0 | [unmannedlab/RELLIS-3D](https://github.com/unmannedlab/RELLIS-3D) |
| RUGD (dataset) | CC BY 4.0 | [rugd.vision](http://rugd.vision/) |
| GOOSE (dataset) | CC BY-SA 4.0 | [goose-dataset.de](https://goose-dataset.de/) |

**Important**: RELLIS-3D uses CC BY-NC-SA 3.0, which prohibits commercial use. Model weights trained with RELLIS-3D inherit this restriction. For commercial use, train with RUGD + GOOSE only.

---

## References

```bibtex
@article{hong2021deep,
  title={Deep Dual-resolution Networks for Real-time and Accurate Semantic Segmentation of Road Scenes},
  author={Hong, Yuanduo and Pan, Huihui and Sun, Weichao and Jia, Yisong},
  journal={arXiv preprint arXiv:2101.06085},
  year={2021}
}
@inproceedings{cai2023efficientvit,
  title={EfficientViT: Lightweight Multi-Scale Attention for High-Resolution Dense Prediction},
  author={Cai, Han and Li, Junyan and Hu, Muyan and Gan, Chuang and Han, Song},
  booktitle={ICCV}, year={2023}
}
@inproceedings{jiang2021rellis3d,
  title={RELLIS-3D Dataset: Data, Benchmarks and Analysis},
  author={Jiang, Peng and Osteen, Philip and Wigness, Maggie and Saripalli, Srikanth},
  booktitle={ICRA}, year={2021}
}
@inproceedings{wigness2019rugd,
  title={A RUGD Dataset for Autonomous Navigation and Visual Perception in Unstructured Outdoor Environments},
  author={Wigness, Maggie and Eum, Sungmin and Rogers, John G and Han, David and Kwon, Heesung},
  booktitle={IROS}, year={2019}
}
@inproceedings{mortimer2024goose,
  title={The GOOSE Dataset for Perception in Unstructured Environments},
  author={Mortimer, Peter and others},
  booktitle={ICRA}, year={2024}
}
```