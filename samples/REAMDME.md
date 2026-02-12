# Inference Samples

Place input files (images or videos) in `input/` and run inference. Results are saved to `output/`.

```
samples/
├── input/           # Place your images or videos here
│   ├── video.mp4
│   ├── frame.jpg
│   └── ...
├── output/          # Results are saved here automatically
│   ├── video_seg.mp4
│   ├── frame_seg.png
│   └── ...
└── README.md
```

---

## Usage

### Video Inference

```bash
# Segmentation only
python scripts/infer_cam.py \
  --checkpoint ./checkpoints/efficientvit-b1/best_model.pth \
  --input ./samples/input/video.mp4

# Overlay mode (original + segmentation blended)
python scripts/infer_cam.py \
  --checkpoint ./checkpoints/efficientvit-b1/best_model.pth \
  --input ./samples/input/video.mp4 \
  --overlay

# Overlay with custom blend ratio (0.0 = original only, 1.0 = segmentation only)
python scripts/infer_cam.py \
  --checkpoint ./checkpoints/efficientvit-b1/best_model.pth \
  --input ./samples/input/video.mp4 \
  --overlay --alpha 0.3

# Save costmap (navigation cost per pixel)
python scripts/infer_cam.py \
  --checkpoint ./checkpoints/efficientvit-b1/best_model.pth \
  --input ./samples/input/video.mp4 \
  --save_costmap
```

### Image Inference

```bash
# Single image
python scripts/infer_cam.py \
  --checkpoint ./checkpoints/efficientvit-b1/best_model.pth \
  --input ./samples/input/frame.jpg

# All images in a directory
python scripts/infer_cam.py \
  --checkpoint ./checkpoints/efficientvit-b1/best_model.pth \
  --input ./samples/input/
```

### Custom Input/Output Paths

Default paths are `./samples/input` and `./samples/output`. Override with `--input` and `--output`:

```bash
python scripts/infer_cam.py \
  --checkpoint ./checkpoints/efficientvit-b1/best_model.pth \
  --input /path/to/any/video.mp4 \
  --output /path/to/save/result.mp4
```

---

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--checkpoint` | (required) | Path to trained model checkpoint (.pth) |
| `--model` | efficientvit-b1 | Model architecture |
| `--input` | ./samples/input | Input image, directory, or video file |
| `--output` | ./samples/output | Output directory or file path |
| `--overlay` | off | Blend segmentation on top of original |
| `--alpha` | 0.5 | Overlay blend ratio (0.0 = original, 1.0 = segmentation) |
| `--save_costmap` | off | Also save navigation costmap |
| `--deploy_size` | 544,640 | Inference resolution H,W |
| `--calibration` | None | Camera calibration .npz file |

---

## Output Modes

**Segmentation (default)**: Each pixel is colored by its predicted class.

**Overlay** (`--overlay`): Original frame blended with segmentation colors. Useful for visually verifying which regions are classified as which class.

**Costmap** (`--save_costmap`): Grayscale image where pixel value represents traversal cost (0 = safe, 255 = impassable). Used for robot path planning.

---

## Class Color Legend

| Class | Color | Traversability |
|-------|-------|----------------|
| Smooth Ground | Purple | Optimal |
| Rough Ground | Brown | Slow down |
| Vegetation | Green | Passable |
| Obstacle | Red | Avoid |
| Water | Blue | Avoid |
| Sky | Cyan | Ignore |
| Dynamic | Yellow | Avoid |

The color legend is automatically appended to video output frames.