import os, subprocess, sys

WORK_DIR = "/mnt/workspace/zhenyong.wzy/work/fd/spatial-ai/hw3/task1_data/objectA_colmap"
IMAGE_DIR = "/mnt/workspace/zhenyong.wzy/work/fd/spatial-ai/hw3/task1_data/objectA_images"

os.makedirs(WORK_DIR, exist_ok=True)
DB_PATH = os.path.join(WORK_DIR, "database.db")
SPARSE_PATH = os.path.join(WORK_DIR, "sparse")

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

cmds = [
    ["colmap", "feature_extractor",
     "--database_path", DB_PATH,
     "--image_path", IMAGE_DIR,
     "--ImageReader.camera_model", "PINHOLE",
     "--ImageReader.single_camera", "1"],
    ["colmap", "exhaustive_matcher",
     "--database_path", DB_PATH],
    ["colmap", "mapper",
     "--database_path", DB_PATH,
     "--image_path", IMAGE_DIR,
     "--output_path", SPARSE_PATH,
     "--Mapper.ba_refine_focal_length", "1",
     "--Mapper.ba_refine_extra_params", "0"],
]

for cmd in cmds:
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"STDERR:\n{result.stderr[-2000:]}")
        sys.exit(1)
    print(f"OK ({len(result.stdout)} chars output)")

print("COLMAP pipeline completed successfully!")
