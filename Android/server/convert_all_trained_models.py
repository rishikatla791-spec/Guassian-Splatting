import os
import shutil
from pathlib import Path
from ply_to_splat import convert_ply_to_splat

SERVER_DIR = Path(__file__).parent.resolve()
MODELS_DIR = SERVER_DIR / "models"
ASSETS_DIR = SERVER_DIR.parent / "app" / "src" / "main" / "assets" / "viewer"

MODELS_DIR.mkdir(exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Convert Train 30k model
train_ply = Path(r"C:\Users\Rishi\Downloads\test\output\pretrained_train\train\point_cloud\iteration_30000\point_cloud.ply")
if train_ply.exists():
    out_splat = MODELS_DIR / "train.splat"
    convert_ply_to_splat(str(train_ply), str(out_splat))
    shutil.copyfile(out_splat, ASSETS_DIR / "demo_train.splat")
    print("Exported demo_train.splat to Android assets!")

# 2. Convert Truck model
truck_ply = Path(r"C:\Users\Rishi\Downloads\test\output\truck\point_cloud\iteration_1000\point_cloud.ply")
if truck_ply.exists():
    out_splat = MODELS_DIR / "truck.splat"
    convert_ply_to_splat(str(truck_ply), str(out_splat))
    shutil.copyfile(out_splat, ASSETS_DIR / "demo_truck.splat")
    print("Exported demo_truck.splat to Android assets!")
