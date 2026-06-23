# Multi-Source 3D Asset Generation & Scene Fusion (3DGS + AIGC) and Cross-Environment Generalization with LeRobot ACT

This repository contains the complete implementation for HW3 of the Spatial AI course, covering two tasks:

- **Task 1**: Multi-source 3D asset generation (multi-view 3DGS, text-to-3D, single-image-to-3D) and fusion with a real-world 3DGS background scene.
- **Task 2**: Training the official LeRobot ACT policy on CALVIN for cross-environment generalization (Env A / A+B+C → Env D).

## Table of Contents

- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Task 1: 3DGS + AIGC Scene Fusion](#task-1-3dgs--aigc-scene-fusion)
- [Task 2: LeRobot ACT Cross-Environment Generalization](#task-2-lerobot-act-cross-environment-generalization)
- [Model Weights](#model-weights)
- [Report](#report)

---

## Environment Setup

### Hardware

- 6–8 × NVIDIA A100-80GB GPUs (DataParallel)
- CUDA 12.1+, Python 3.11

### Install Dependencies

```bash
# Core packages
pip install torch==2.4.1 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install lerobot==0.4.4
pip install pyarrow Pillow matplotlib opencv-python trimesh numpy==1.26.4

# Official 3DGS (Task 1)
pip install git+https://github.com/graphdeco-inria/gaussian-splatting --no-build-isolation
pip install diff-gaussian-rasterization simple-knn --no-build-isolation

# threestudio (Task 1 AIGC) — clone and install separately
git clone https://github.com/threestudio/threestudio.git
cd threestudio && pip install -e .
```

See `requirements.txt` for the full pinned dependency list.

---

## Data Preparation

### Task 1: Multi-view Photos + Mip-NeRF 360

1. **Object A photos**: Capture a smartphone video around a real object, extract frames:
   ```bash
   ffmpeg -i video.mp4 -vf "fps=5" frames/frame_%04d.jpg
   ```
   Place in `data/objectA_images/`.

2. **Mip-NeRF 360 garden**: Download from [the official site](https://jonbarron.info/mipnerf360/):
   ```bash
   wget https://storage.googleapis.com/gresearch/refraw360/360_v2/garden.zip
   unzip garden.zip -d data/garden/
   ```

3. **Object C input image**: A single RGBA image (background removed). An example is provided in `data/cactus_rgba.png`.

### Task 2: CALVIN Dataset (LeRobot format)

```bash
bash scripts/task2_act/download_calvin.sh
```

This downloads the CALVIN dataset in LeRobot v2.1 format (~66 GB) to `data/calvin-lerobot/` with splits A, B, C, D.

---

## Task 1: 3DGS + AIGC Scene Fusion

### Step 1: Object A — Multi-View 3DGS

```bash
# 1a. COLMAP sparse reconstruction
python scripts/task1_3dgs/run_colmap_objectA.py \
    --image_dir data/objectA_images \
    --output_dir data/objectA_colmap

# 1b. Train official 3DGS (30K iterations)
python -m gaussian_splatting.train \
    -s data/objectA_colmap \
    -m outputs/objectA_3dgs \
    --iterations 30000
```

### Step 2: Background — Garden 3DGS

```bash
python -m gaussian_splatting.train \
    -s data/garden \
    -m outputs/garden_3dgs \
    --iterations 30000
```

### Step 3: Object B — Text-to-3D (Fantasia3D)

```bash
# Geometry stage (10K steps)
python threestudio/train.py \
    --config configs/fantasia3d-pineapple.yaml \
    --train guidance.scale 30 \
    --max_steps 10000

# Texture stage (5K steps, frozen geometry)
python threestudio/train.py \
    --config configs/fantasia3d-pineapple-texture.yaml \
    --max_steps 50000

# Export mesh
bash scripts/task1_aigc/export_objectB.sh
```

### Step 4: Object C — Single-Image-to-3D (Zero123)

```bash
# Train Stable Zero123 (3K steps)
bash scripts/task1_aigc/run_objectC.sh

# Export and clean mesh
python scripts/task1_aigc/extract_mesh.py
python scripts/task1_aigc/render_mesh_clean.py
```

### Step 5: Scene Fusion & Flythrough

```bash
# Fuse all objects into the garden 3DGS scene
python scripts/task1_fusion/fuse_scene_v2.py \
    --garden outputs/garden_3dgs \
    --pot outputs/objectA_3dgs \
    --pineapple outputs/objectB_mesh.obj \
    --cactus outputs/objectC_mesh.obj \
    --output outputs/fused_scene.ply

# Render flythrough video
python scripts/task1_fusion/render_flythrough.py \
    --scene outputs/fused_scene.ply \
    --output outputs/fusion_flythrough.mp4
```

---

## Task 2: LeRobot ACT Cross-Environment Generalization

### Training

```bash
# Env A only (basic policy)
python scripts/task2_act/train_act_official.py \
    --envs A \
    --eval-envs D \
    --epochs 200 \
    --batch-size 256 \
    --lr 1e-4 \
    --chunk-size 30 \
    --output-dir outputs/envA

# Env A+B+C (multi-environment)
python scripts/task2_act/train_act_official.py \
    --envs ABC \
    --eval-envs D \
    --epochs 200 \
    --batch-size 256 \
    --lr 1e-4 \
    --chunk-size 30 \
    --output-dir outputs/envABC
```

### Chunk Size Experiments

```bash
bash scripts/task2_act/run_chunk_experiments.sh
```

### Evaluation

```bash
# Main results: per-dimension L1 and success rates
python scripts/task2_act/eval_act_official.py

# Chunk comparison
python scripts/task2_act/eval_chunk_experiments.py
```

### Generate Figures

```bash
python scripts/task2_act/gen_act_official_fig.py
```

---

## Model Weights

Pre-trained model weights are included in `checkpoints/`:

| Path | Size | Description |
|------|------|-------------|
| `checkpoints/task1/objectA_3dgs/` | 231 MB | Flower pot 3DGS (official, PSNR 31.94) |
| `checkpoints/task1/garden_3dgs/` | 1.0 GB | Mip-NeRF 360 garden 3DGS (official, PSNR 29.65) |
| `checkpoints/task1/objectB_pineapple/` | 5 MB | Pineapple mesh (OBJ + PBR textures) |
| `checkpoints/task1/objectC_cactus/` | 1 MB | Cactus mesh (OBJ + baked PLY) |
| `checkpoints/task1/fused_scene.ply` | 26 MB | Final fused scene (4.5M Gaussians) |
| `checkpoints/task2/envA_epoch199.pt` | 197 MB | ACT checkpoint (Env A, epoch 199) |
| `checkpoints/task2/envABC_epoch199.pt` | 197 MB | ACT checkpoint (Env A+B+C, epoch 199) |
| `checkpoints/task2/envA_chunk10_epoch199.pt` | 197 MB | ACT checkpoint (chunk=10, epoch 199) |
| `checkpoints/task2/envA_chunk50_epoch199.pt` | 198 MB | ACT checkpoint (chunk=50, epoch 199) |

For 3DGS rendering, each model directory includes `cameras.json`, `cfg_args`, and `exposure.json` metadata required by the official gaussian-splatting viewer.

---

## Report

The LaTeX source and compiled PDF are in `report/`:

```bash
cd report
pdflatex main.tex && pdflatex main.tex
```

---

## Repository Structure

```
.
├── report/                  # LaTeX report + figures + PDF
├── scripts/
│   ├── task1_3dgs/          # COLMAP + 3DGS training scripts
│   ├── task1_aigc/          # Fantasia3D + Zero123 scripts
│   ├── task1_fusion/        # Scene fusion + flythrough rendering
│   ├── task2_act/           # ACT training + evaluation
│   └── figures/             # Report figure generation
├── configs/                 # threestudio YAML configs
├── checkpoints/             # Model weights (3DGS + mesh + ACT)
│   ├── task1/               # Object A/B/C + garden + fused scene
│   └── task2/               # ACT checkpoints (4 models, final epoch)
├── outputs/                 # Flythrough video, eval results, training logs
├── data/                    # Input data (cactus_rgba.png) + download guide
├── requirements.txt
└── README.md
```

## Citation

If you use this work, please cite:

```bibtex
@misc{spatial-ai-hw3,
  author = {Zhenyong Wei},
  title = {Multi-Source 3D Asset Generation and Scene Fusion via 3DGS, and Cross-Environment Generalization with LeRobot ACT},
  year = {2026},
  url = {https://github.com/ZhenYongWei/sp-hw3}
}
```
