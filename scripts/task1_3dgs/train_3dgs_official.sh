#!/bin/bash
# Train official 3DGS (graphdeco-inria/gaussian-splatting)
# This produces the final high-quality models reported in the paper.
#
# Prerequisites:
#   pip install git+https://github.com/graphdeco-inria/gaussian-splatting --no-build-isolation
#   (compiles diff-gaussian-rasterization and simple-knn CUDA extensions)
#
# Usage:
#   bash train_3dgs_official.sh <scene_type> <data_dir> <output_dir>
#   scene_type: "object" (single object) or "garden" (large outdoor scene)

set -e

SCENE_TYPE="${1:-garden}"
DATA_DIR="${2:-data/garden}"
OUTPUT_DIR="${3:-outputs/garden_3dgs}"

echo "=== Official 3DGS Training ==="
echo "Scene: $SCENE_TYPE"
echo "Data:  $DATA_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

if [ "$SCENE_TYPE" = "garden" ]; then
    # Garden: large outdoor scene, ~1.6K resolution
    python -m gaussian_splatting.train \
        -s "$DATA_DIR" \
        -m "$OUTPUT_DIR" \
        --iterations 30000 \
        --resolution 1 \
        --data_device cuda \
        --eval
elif [ "$SCENE_TYPE" = "object" ]; then
    # Object A: single object, 640x1129, SIMPLE_RADIAL camera model
    python -m gaussian_splatting.train \
        -s "$DATA_DIR" \
        -m "$OUTPUT_DIR" \
        --iterations 30000 \
        --resolution 1 \
        --data_device cuda \
        --eval
else
    echo "Unknown scene type: $SCENE_TYPE (use 'object' or 'garden')"
    exit 1
fi

echo ""
echo "=== Training complete ==="
echo "Model saved to: $OUTPUT_DIR/point_cloud/iteration_30000/point_cloud.ply"
echo ""
echo "Render test views:"
python -m gaussian_splatting.render \
    -m "$OUTPUT_DIR"

echo ""
echo "Compute metrics:"
python -m gaussian_splatting.metrics \
    -m "$OUTPUT_DIR"
