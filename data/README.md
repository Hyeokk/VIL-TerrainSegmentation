# Dataset Directory Structure

This directory holds all training datasets. Three open-source off-road
datasets are combined into a unified 7-class traversability ontology.

RELLIS-3D is **required**. RUGD and GOOSE are optional but recommended
for better generalization across diverse terrain types.


## Quick Setup

```bash
# 1. Download and unpack each dataset (see links below)
# 2. Create custom 70/30 split for RELLIS-3D
python scripts/make_split_custom.py

# 3. (Recommended) Pre-resize for faster training
python scripts/preprocess_datasets.py

# 4. Verify
python scripts/verify_all.py
```


## Expected Directory Layout

```
data/
├── README.md                          (this file)
│
├── Rellis-3D/                         [REQUIRED] 4,169 train images
│   ├── 00000/
│   │   ├── pylon_camera_node/         RGB images (1200x1920 PNG)
│   │   │   ├── frame000000-1581624652_750.jpg
│   │   │   └── ...
│   │   └── pylon_camera_node_label_id/  Annotation masks (class ID per pixel)
│   │       ├── frame000000-1581624652_750.png
│   │       └── ...
│   ├── 00001/
│   ├── 00002/
│   ├── 00003/
│   ├── 00004/
│   ├── split/                         Official splits (downloaded with dataset)
│   │   ├── train.lst
│   │   ├── val.lst
│   │   └── test.lst
│   └── split_custom/                  Custom 70/30 split (generated)
│       ├── train_70.lst
│       └── test_30.lst
│
├── RUGD/                              [OPTIONAL] 7,436 train images
│   ├── RUGD_frames-with-annotations/  RGB images (550x688 PNG)
│   │   ├── creek/
│   │   │   ├── creek_00000.png
│   │   │   └── ...
│   │   ├── park-1/
│   │   ├── park-2/
│   │   ├── park-8/
│   │   ├── trail/
│   │   ├── trail-3/
│   │   ├── trail-4/
│   │   ├── trail-5/
│   │   ├── trail-6/
│   │   ├── trail-7/
│   │   ├── trail-9/
│   │   ├── trail-10/
│   │   ├── trail-11/
│   │   ├── trail-12/
│   │   ├── trail-13/
│   │   ├── trail-14/
│   │   ├── trail-15/
│   │   ├── village/
│   │   └── ...
│   └── RUGD_annotations/             Color-coded annotation PNGs
│       ├── creek/
│       │   ├── creek_00000.png
│       │   └── ...
│       ├── park-1/
│       └── ...                        (same scene names as frames)
│
├── GOOSE/                             [OPTIONAL] 7,845 train images
│   ├── goose_label_mapping.csv        Class ID-to-name mapping (64 classes)
│   ├── images/
│   │   └── train/
│   │       ├── <scene_name>/
│   │       │   ├── <prefix>_windshield_vis.png   (visible RGB)
│   │       │   └── ...
│   │       └── ...
│   └── labels/
│       └── train/
│           ├── <scene_name>/
│           │   ├── <prefix>_labelids.png          (class ID per pixel)
│           │   └── ...
│           └── ...
│
├── Rellis-3D_fast/                    [GENERATED] Pre-resized RELLIS (max 1024px, JPEG)
├── RUGD_fast/                         [GENERATED] Pre-resized RUGD
└── GOOSE_fast/                        [GENERATED] Pre-resized GOOSE
```


## Dataset Details

### RELLIS-3D (Required)

| Item | Value |
|------|-------|
| Images | 6,235 total (4,169 train / 1,788 test after 70/30 split) |
| Resolution | 1200 x 1920 (H x W) |
| Format | PNG (images), PNG (ID labels) |
| Classes | 20 original -> 7 unified |
| Environment | US Army test facility, off-road trails |
| Source | [unmannedlab/RELLIS-3D](https://github.com/unmannedlab/RELLIS-3D) |

**Split file format** (space-separated relative paths):
```
00000/pylon_camera_node/frame000000-1581624652_750.jpg 00000/pylon_camera_node_label_id/frame000000-1581624652_750.png
00000/pylon_camera_node/frame000001-1581624652_950.jpg 00000/pylon_camera_node_label_id/frame000001-1581624652_950.png
...
```

**Download:**
1. Go to [RELLIS-3D GitHub](https://github.com/unmannedlab/RELLIS-3D)
2. Download "Images" (camera images) and "ID Annotations" (semantic labels)
3. Download "Split files" (train/val/test lists)
4. Unpack so that sequences (00000-00004) are directly under `data/Rellis-3D/`

### RUGD (Optional)

| Item | Value |
|------|-------|
| Images | 7,436 |
| Resolution | 550 x 688 (H x W) |
| Format | PNG (images), PNG (color-coded labels) |
| Classes | 24 original -> 7 unified |
| Environment | Parks, trails, forests |
| Source | [RUGD Dataset](http://rugd.vision/) |

**Note:** RUGD uses color-encoded annotation PNGs (each pixel is an RGB color
representing a class), not integer class IDs. The dataloader converts colors
to unified class IDs automatically.

**Download:**
1. Go to [RUGD website](http://rugd.vision/)
2. Download "RUGD_frames-with-annotations" and "RUGD_annotations"
3. Unpack both into `data/RUGD/`

### GOOSE (Optional)

| Item | Value |
|------|-------|
| Images | 7,845 |
| Resolution | Varies by scene |
| Format | PNG (images: `*_windshield_vis.png`), PNG (labels: `*_labelids.png`) |
| Classes | 64 original -> 7 unified |
| Environment | European outdoor/forest, diverse terrain |
| Source | [GOOSE Dataset](https://goose-dataset.de/) |

**Supported directory layouts** (auto-detected):

Layout A (official):
```
data/GOOSE/images/train/<scene>/*_windshield_vis.png
data/GOOSE/labels/train/<scene>/*_labelids.png
```

Layout B (single archive):
```
data/GOOSE/train/images/train/<scene>/*_windshield_vis.png
data/GOOSE/train/labels/train/<scene>/*_labelids.png
```

**Download:**
1. Go to [GOOSE website](https://goose-dataset.de/)
2. Download the training split images and labels
3. Ensure `goose_label_mapping.csv` is present (included with dataset)
4. Unpack into `data/GOOSE/`


## Fast Mode (Pre-resized)

Running `python scripts/preprocess_datasets.py` creates `_fast` directories
with images resized to max 1024px (JPEG quality 95) and labels resized with
nearest-neighbor interpolation to preserve class IDs.

| Item | Original | Fast |
|------|----------|------|
| Format | PNG | JPEG (quality=95) |
| Max resolution | up to 1920px | max 1024px |
| File size | ~5MB | ~100KB |
| Decode time | ~50ms | ~5ms |
| Epoch time | ~472s | ~188s |

Labels remain PNG to avoid interpolation artifacts.

Use `--fast` flag during training:
```bash
python scripts/train.py --model ddrnet23-slim --fast --num_workers 8
```


## Unified 7-Class Ontology

All three datasets are remapped to a common 7-class system designed for
caterpillar-track robot traversability:

| ID | Class | Description | Navigation |
|----|-------|-------------|------------|
| 0 | Smooth Ground | Asphalt, concrete, packed dirt, cobblestone | Optimal path |
| 1 | Rough Ground | Sand, gravel, mud, snow, rubble | Passable (slow down) |
| 2 | Vegetation | Low grass, moss, fallen leaves | Passable (caterpillar) |
| 3 | Obstacle | Trees, rocks, buildings, fences, bushes | Avoid |
| 4 | Water | Puddles, streams, lakes | Avoid (flood risk) |
| 5 | Sky | Sky, clouds | Ignore |
| 6 | Dynamic | People, vehicles, bicycles, animals | Avoid (safety) |
| 255 | ignore | Void, undefined, ego vehicle | Excluded from loss |

**Caterpillar-specific mappings:**
- bush -> Obstacle (ID 3): Dense bushes can entangle caterpillar tracks
- puddle -> Water (ID 4): Unknown depth, risk of drivetrain flooding
- dirt -> Smooth Ground (ID 0): Packed dirt is optimal terrain for tracks
- grass -> Vegetation (ID 2): Low grass is traversable but distinct from ground