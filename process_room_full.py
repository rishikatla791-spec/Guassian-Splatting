"""
Full End-to-End Smartphone Room Pipeline:
1. COLMAP SfM (GPU Feature Extraction, Matching, Sparse Reconstruction, Undistortion)
2. 3D Gaussian Splatting Training (3,000 steps on RTX 3050 GPU)
3. Auto-load into Viser 3D Visualizer & Open Browser
"""

import os
import sys
import time
import subprocess
import shutil

COLMAP_BAT = r"C:\tools\COLMAP\COLMAP-3.9.1-windows-cuda\COLMAP.bat"
DATASET_DIR = r"C:\Users\Rishi\Downloads\test\Room_dataset"
OUTPUT_DIR = r"C:\Users\Rishi\Downloads\test\output\room"
PYTHON_EXE = r"C:\Users\Rishi\anaconda3\envs\gaussian_cuda\python.exe"

def run_cmd(cmd, step_name):
    print(f"\n[{step_name}] Running command:\n  {cmd}\n", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(f"[Error] {step_name} failed with exit code {res.returncode}", flush=True)
        sys.exit(res.returncode)
    print(f"[{step_name}] Completed in {time.time() - t0:.2f}s\n", flush=True)

print("=" * 65)
print("  STARTING END-TO-END ROOM 3D RECONSTRUCTION PIPELINE")
print("=" * 65)

# Step 1: COLMAP Setup
db_path = os.path.join(DATASET_DIR, "distorted", "database.db")
distorted_sparse = os.path.join(DATASET_DIR, "distorted", "sparse")
input_images = os.path.join(DATASET_DIR, "input")

os.makedirs(distorted_sparse, exist_ok=True)
if os.path.exists(db_path):
    os.remove(db_path)

# 1. Feature Extraction (CUDA SIFT)
feat_cmd = (
    f'"{COLMAP_BAT}" feature_extractor '
    f'--database_path "{db_path}" '
    f'--image_path "{input_images}" '
    f'--ImageReader.single_camera 0 '
    f'--ImageReader.camera_model SIMPLE_RADIAL '
    f'--SiftExtraction.use_gpu 1'
)
run_cmd(feat_cmd, "1/4 COLMAP Feature Extraction (GPU)")

# 2. Feature Matching (GPU Exhaustive Matcher)
match_cmd = (
    f'"{COLMAP_BAT}" exhaustive_matcher '
    f'--database_path "{db_path}" '
    f'--SiftMatching.use_gpu 1'
)
run_cmd(match_cmd, "2/4 COLMAP GPU Feature Matching")

# 3. 3D Sparse Mapper
map_cmd = (
    f'"{COLMAP_BAT}" mapper '
    f'--database_path "{db_path}" '
    f'--image_path "{input_images}" '
    f'--output_path "{distorted_sparse}" '
    f'--Mapper.ba_global_function_tolerance=0.000001'
)
run_cmd(map_cmd, "3/4 COLMAP 3D Bundle Adjustment Mapper")

# 4. Image Undistortion
undist_cmd = (
    f'"{COLMAP_BAT}" image_undistorter '
    f'--image_path "{input_images}" '
    f'--input_path "{distorted_sparse}\\0" '
    f'--output_path "{DATASET_DIR}" '
    f'--output_type COLMAP'
)
run_cmd(undist_cmd, "4/4 COLMAP Image Undistortion")

# Ensure sparse/0 structure matches 3DGS requirements
sparse_root = os.path.join(DATASET_DIR, "sparse")
sparse_zero = os.path.join(DATASET_DIR, "sparse", "0")
os.makedirs(sparse_zero, exist_ok=True)
for item in os.listdir(sparse_root):
    if item == "0":
        continue
    src = os.path.join(sparse_root, item)
    dst = os.path.join(sparse_zero, item)
    if os.path.isfile(src):
        shutil.move(src, dst)

print("\n" + "=" * 65)
print("  COLMAP RECONSTRUCTION COMPLETE! STARTING 3DGS TRAINING...")
print("=" * 65 + "\n")

# Step 2: 3D Gaussian Splatting Training
train_script = r"C:\Users\Rishi\Downloads\test\gaussian-splatting\train.py"
train_cmd = (
    f'call "C:\\Users\\Rishi\\Downloads\\test\\build_env.bat" "{PYTHON_EXE}" "{train_script}" '
    f'-s "{DATASET_DIR}" -m "{OUTPUT_DIR}" --eval --iterations 3000 --save_iterations 1000 2000 3000 --resolution 2'
)
run_cmd(train_cmd, "3D Gaussian Splatting Training")

print("\n" + "=" * 65)
print("  TRAINING COMPLETE! LAUNCHING VISER 3D VISUALIZER...")
print("=" * 65 + "\n")

# Step 3: Launch Viser Viewer
viser_script = r"C:\Users\Rishi\Downloads\test\viser_viewer.py"
subprocess.Popen([PYTHON_EXE, viser_script])
print(f"Viser visualizer running at http://localhost:8080")
