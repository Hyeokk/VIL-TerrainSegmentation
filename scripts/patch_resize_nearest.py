#!/usr/bin/env python3
"""
Convert all Resize nodes in DDRNet ONNX from bilinear to nearest mode.

QAIRT 2.40 HTP supports Resize with mode='nearest' but NOT mode='linear'.
This script patches all Resize nodes so the entire model can run on NPU.

For segmentation (argmax-based), nearest vs bilinear quality difference
is negligible.

Usage (host PC):
    python scripts/patch_resize_nearest.py \
        --input deploy/ddrnet23_slim_unified7class_544x640.onnx \
        --output deploy/ddrnet23_slim_unified7class_544x640_nearest.onnx
"""

import os
import argparse
import onnx


def patch_resize_nodes(model):
    """Change all Resize nodes from bilinear/linear to nearest."""
    resize_nodes = [n for n in model.graph.node if n.op_type == "Resize"]
    print(f"[Patch] Found {len(resize_nodes)} Resize node(s)")

    for i, node in enumerate(resize_nodes):
        old_mode = "unknown"
        old_ctm = "unknown"

        # Read current attributes
        for attr in node.attribute:
            if attr.name == "mode":
                old_mode = attr.s.decode() if isinstance(attr.s, bytes) else attr.s
            if attr.name == "coordinate_transformation_mode":
                old_ctm = attr.s.decode() if isinstance(attr.s, bytes) else attr.s

        # Remove existing mode and coordinate_transformation_mode
        attrs_to_remove = []
        for attr in node.attribute:
            if attr.name in ("mode", "coordinate_transformation_mode"):
                attrs_to_remove.append(attr)
        for attr in attrs_to_remove:
            node.attribute.remove(attr)

        # Set nearest mode + asymmetric (HTP compatible)
        node.attribute.append(onnx.helper.make_attribute("mode", "nearest"))
        node.attribute.append(
            onnx.helper.make_attribute(
                "coordinate_transformation_mode", "asymmetric"
            )
        )

        print(f"  [{i+1}] {node.name or node.output[0]}")
        print(f"      mode: {old_mode} -> nearest")
        print(f"      ctm:  {old_ctm} -> asymmetric")

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Patch Resize nodes to nearest mode for HTP"
    )
    parser.add_argument(
        "--input", type=str,
        default="deploy/ddrnet23_slim_unified7class_544x640.onnx",
    )
    parser.add_argument(
        "--output", type=str,
        default="deploy/ddrnet23_slim_unified7class_544x640_nearest.onnx",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Patch Resize: bilinear -> nearest")
    print(f"{'='*60}")
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}")

    model = onnx.load(args.input)
    model = patch_resize_nodes(model)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    onnx.save(model, args.output)

    # Verify
    import onnxruntime as ort
    import numpy as np

    sess = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    size_mb = os.path.getsize(args.output) / (1024 * 1024)

    print(f"\n[Verify]")
    print(f"  Input:  {inp.name} {inp.shape}")
    print(f"  Output: {out.name} {out.shape}")
    print(f"  Size:   {size_mb:.1f} MB")

    dummy = np.random.randn(*inp.shape).astype(np.float32)
    result = sess.run(None, {inp.name: dummy})[0]
    print(f"  Test output shape: {result.shape}")
    print(f"  PASSED")

    # Check all Resize nodes are now nearest
    ops_check = []
    for node in model.graph.node:
        if node.op_type == "Resize":
            for attr in node.attribute:
                if attr.name == "mode":
                    mode = attr.s.decode() if isinstance(attr.s, bytes) else attr.s
                    ops_check.append(mode)
    print(f"  Resize modes: {ops_check}")

    print(f"\n[Next Steps]")
    print(f"  1. Quantize (host PC):")
    print(f"     python scripts/quantize_onnx.py \\")
    print(f"         --onnx_model {args.output} \\")
    print(f"         --output deploy/ddrnet23_slim_int8_qdq_nearest.onnx \\")
    print(f"         --num_calibration 200")
    print(f"")
    print(f"  2. Transfer to IQ-9075:")
    print(f"     scp deploy/ddrnet23_slim_int8_qdq_nearest.onnx \\")
    print(f"         ubuntu@<IP>:~/VIL-Project-AMR/amr_segmentation/models/")
    print(f"")
    print(f"  3. Run on IQ-9075:")
    print(f"     python3 scripts/infer_video.py \\")
    print(f"         --model models/ddrnet23_slim_int8_qdq_nearest.onnx \\")
    print(f"         --input samples/inputs/road.mp4 \\")
    print(f"         --output samples/outputs/road_result.mp4 --overlay")


if __name__ == "__main__":
    main()