# Data Preparation Guide

## Task 1 Data

### Object A: Smartphone Photos

1. Record a video circling a real object (flower pot) at ~30 fps for ~20 seconds
2. Extract frames:
   ```bash
   mkdir -p objectA_images
   ffmpeg -i video.mp4 -vf "fps=5,scale=640:-1" objectA_images/frame_%04d.jpg
   ```
3. Run COLMAP:
   ```bash
   python ../scripts/task1_3dgs/run_colmap_objectA.py \
       --image_dir objectA_images \
       --output_dir objectA_colmap
   ```

### Garden Background: Mip-NeRF 360

Download from the [official site](https://jonbarron.info/mipnerf360/):
```bash
wget https://storage.googleapis.com/gresearch/refraw360/360_v2/garden.zip
unzip garden.zip -d garden/
```

### Object C: Single RGBA Image

A pre-segmented image (`cactus_rgba.png`) is included in this directory.
For custom objects, remove the background using `rembg`:
```bash
pip install rembg
rembg i input.jpg output_rgba.png
```

## Task 2 Data: CALVIN

```bash
# Download CALVIN in LeRobot v2.1 format (~66 GB)
# From the repository root:
bash ../scripts/task2_act/download_calvin.sh
```

Expected structure:
```
data/
├── calvin-lerobot/
│   ├── splitA/
│   │   └── data/chunk-*/episode_*.parquet
│   ├── splitB/
│   ├── splitC/
│   └── splitD/
├── garden/              # Mip-NeRF 360
├── objectA_images/      # Phone photos
├── objectA_colmap/      # COLMAP output
└── cactus_rgba.png      # Zero123 input
```

## Model Weights

Download pre-trained checkpoints from cloud storage (see main README.md).
